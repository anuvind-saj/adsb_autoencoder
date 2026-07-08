"""
extract_labels.py — One-time preprocessing: extract supervised training pairs
             from real-world .npy IQ burst files.

What it does
------------
For each .npy burst file captured by adsb_iq_sample_collection:

  1. Slide a preamble detector across the raw magnitude signal.
  2. For each candidate position, check the preamble shape and validate the
     112-bit payload with pyModeS CRC.
  3. For each CRC-passing frame, estimate the signal's amplitude, initial
     carrier phase, and frequency offset from the real preamble pulses.
  4. Re-synthesize a mathematically perfect "clean" IQ target using
     generator.py with those estimated parameters.
  5. Scale the clean target so its RMS power matches the real window.
  6. Save pairs as compressed NumPy arrays (.npz) — one file per burst.

Output files
------------
Each input  <burst_name>.npy   produces  <burst_name>_labels.npz containing:

  noisy_iq   : float32  (N, 2, 240) — real captured IQ windows
  clean_iq   : float32  (N, 2, 240) — re-synthesized clean IQ targets
  hex_msgs   : str list  length N   — decoded hex messages (upper-case)
  positions  : int32    (N,)        — sample offset within the burst file
  amp_est    : float32  (N,)        — estimated carrier amplitudes
  f_offset_est: float32 (N,)        — estimated frequency offsets (Hz)
  phase_est  : float32  (N,)        — estimated initial phases (rad)

Usage
-----
# Extract labels from all .npy files in a directory (recursive):
python extract_labels.py \\
    --data-dir  ~/adsb_iq_data \\
    --labels-dir ~/adsb_iq_data/labels \\
    --recursive

# Limit to 10 files for a quick check:
python extract_labels.py \\
    --data-dir  ~/adsb_iq_data \\
    --labels-dir ~/adsb_iq_data/labels \\
    --max-files 10 \\
    --verbose

# Skip files whose label .npz already exists (safe to re-run):
python extract_labels.py \\
    --data-dir  ~/adsb_iq_data \\
    --labels-dir ~/adsb_iq_data/labels \\
    --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match generator.py and model.py)
# ---------------------------------------------------------------------------

SAMPLE_RATE: float = 2.0e6
SAMPLE_PERIOD: float = 1.0 / SAMPLE_RATE   # 0.5 µs
FRAME_SAMPLES: int = 240
PREAMBLE_SAMPLES: int = 16
DATA_SAMPLES: int = 224                     # 112 bits × 2 samples/bit
PREAMBLE_PULSE_IDX: Tuple[int, ...] = (0, 2, 7, 9)   # high samples
PREAMBLE_LOW_IDX: Tuple[int, ...] = (1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15)

# ---------------------------------------------------------------------------
# Optional pyModeS import
# ---------------------------------------------------------------------------

try:
    from pyModeS.util import crc as _pms_crc
    def _crc(hex_msg: str) -> int:
        return _pms_crc(hex_msg)
    HAS_PYMODES = True
except Exception:
    HAS_PYMODES = False
    def _crc(hex_msg: str) -> int:  # type: ignore[misc]
        raise RuntimeError(
            "pyModeS is required for CRC validation.  "
            "Install it with:  pip install pyModeS"
        )


# ---------------------------------------------------------------------------
# Signal parameter estimation from real preamble
# ---------------------------------------------------------------------------

def _estimate_signal_params(
    I_win: np.ndarray,
    Q_win: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Estimate carrier amplitude, initial phase, and frequency offset from the
    four preamble pulses in a real 240-sample IQ window.

    Returns
    -------
    amplitude   : Mean magnitude at the preamble pulse positions.
    initial_phase : Carrier phase at sample 0 (first preamble pulse), radians.
    f_offset_hz : Estimated carrier frequency offset in Hz.

    Method
    ------
    Phase at each pulse position p:  φ(p) = atan2(Q[p], I[p])
    For a linear phase model φ(n) = 2π·f·n·Ts + φ₀:

        From pulses at n=0 and n=2:
            Δφ₀₂ = φ(2) - φ(0)  ≈ 2π · f · 2 · Ts

        From pulses at n=7 and n=9:
            Δφ₇₉ = φ(9) - φ(7)  ≈ 2π · f · 2 · Ts

    We average the two estimates for robustness.

    Phase wrapping is handled with np.angle(exp(j·Δφ)).
    """
    pulse_I = I_win[list(PREAMBLE_PULSE_IDX)]
    pulse_Q = Q_win[list(PREAMBLE_PULSE_IDX)]

    # Amplitude: mean magnitude across all four pulse positions
    amplitude = float(np.sqrt(pulse_I ** 2 + pulse_Q ** 2).mean()) + 1e-9

    # Phase at each pulse
    phi = np.arctan2(pulse_Q, pulse_I)   # phi[0..3] = phases at indices 0,2,7,9

    # Frequency estimate from phase differences (two independent pairs)
    delta_n = 2  # both pairs span 2 samples
    dphi_02 = float(np.angle(np.exp(1j * (phi[1] - phi[0]))))  # wraps safely
    dphi_79 = float(np.angle(np.exp(1j * (phi[3] - phi[2]))))
    dphi_avg = 0.5 * (dphi_02 + dphi_79)
    f_offset_hz = dphi_avg / (2.0 * np.pi * delta_n * SAMPLE_PERIOD)

    # Initial phase: phase at sample 0, corrected back from f_offset rotation
    # φ₀ = φ(0) - 2π·f·0·Ts = φ(0)
    initial_phase = float(phi[0])

    return amplitude, initial_phase, f_offset_hz


# ---------------------------------------------------------------------------
# Single-file label extractor
# ---------------------------------------------------------------------------

def extract_file(
    npy_path: Path,
    preamble_thr_percentile: float = 90.0,
    preamble_thr_multiplier: float = 3.5,
    preamble_thr_min: float = 0.05,
    preamble_low_ratio: float = 3.0,
    verbose: bool = False,
) -> Dict:
    """
    Extract all valid ADS-B frames from a single .npy burst file.

    Parameters
    ----------
    npy_path                : Path to the complex64 .npy burst file.
    preamble_thr_percentile : Percentile of magnitude used for noise floor.
    preamble_thr_multiplier : Noise-floor multiplier for the high-sample threshold.
    preamble_thr_min        : Absolute minimum threshold (avoids false positives
                              in very low-amplitude captures).
    preamble_low_ratio      : ``high_mean > low_mean * ratio`` must hold.
    verbose                 : Print per-file statistics.

    Returns
    -------
    dict with keys:
        noisy_iq    : (N, 2, 240) float32
        clean_iq    : (N, 2, 240) float32
        hex_msgs    : list[str]   length N
        positions   : (N,) int32
        amp_est     : (N,) float32
        f_offset_est: (N,) float32
        phase_est   : (N,) float32
        stats       : dict  (diagnostics)
    """
    # Import here to avoid circular imports at module level
    from generator import ADSBSignalGenerator, SignalParams

    arr = np.load(npy_path, mmap_mode="r")
    I_full = arr.real.astype(np.float32)
    Q_full = arr.imag.astype(np.float32)

    # Remove DC offset (RTL-SDR hardware bias)
    I_full = I_full - float(I_full.mean())
    Q_full = Q_full - float(Q_full.mean())

    mag = np.sqrt(I_full ** 2 + Q_full ** 2)
    n_samples = len(mag)

    # ── Adaptive preamble threshold ──────────────────────────────────────────
    noise_floor = float(np.percentile(mag, preamble_thr_percentile))
    thr = max(noise_floor * preamble_thr_multiplier, preamble_thr_min)

    # ── Vectorised preamble search ───────────────────────────────────────────
    # Need at least 16 samples for the preamble window
    if n_samples < PREAMBLE_SAMPLES:
        return _empty_result({"error": "file too short"})

    views = np.lib.stride_tricks.sliding_window_view(mag, PREAMBLE_SAMPLES)
    high = views[:, list(PREAMBLE_PULSE_IDX)].mean(axis=1)
    low  = views[:, list(PREAMBLE_LOW_IDX)].mean(axis=1)
    candidates = np.where((high > thr) & (high > low * preamble_low_ratio))[0]

    if verbose:
        print(f"  {npy_path.name}: noise_floor={noise_floor:.4f}  "
              f"thr={thr:.4f}  candidates={len(candidates)}")

    # ── Per-candidate validation ─────────────────────────────────────────────
    gen = ADSBSignalGenerator(sample_rate=SAMPLE_RATE)

    noisy_list: List[np.ndarray] = []
    clean_list: List[np.ndarray] = []
    hex_list:   List[str] = []
    pos_list:   List[int] = []
    amp_list:   List[float] = []
    foff_list:  List[float] = []
    phi_list:   List[float] = []

    prev_valid_pos = -FRAME_SAMPLES  # enforce minimum spacing between frames

    for pos in candidates:
        end = pos + FRAME_SAMPLES
        if end > n_samples:
            break
        if pos < prev_valid_pos + FRAME_SAMPLES:
            continue   # too close to last valid frame (overlap guard)

        I_win = I_full[pos:end]
        Q_win = Q_full[pos:end]

        # ── Decode PPM bits ──────────────────────────────────────────────────
        # Data region starts at sample PREAMBLE_SAMPLES (index 16)
        data_mag = mag[pos + PREAMBLE_SAMPLES: end]
        bits_raw = data_mag.reshape(112, 2)
        bits = (bits_raw[:, 0] > bits_raw[:, 1]).astype(np.uint8)
        hex_msg = np.packbits(bits).tobytes().hex().upper()

        # ── CRC check ───────────────────────────────────────────────────────
        try:
            crc_val = _crc(hex_msg)
        except Exception:
            continue
        if crc_val != 0:
            continue

        # ── Estimate signal parameters ───────────────────────────────────────
        amp, phi0, f_off = _estimate_signal_params(I_win, Q_win)

        # ── Re-synthesize clean target ────────────────────────────────────────
        params = SignalParams(
            amplitude=amp,
            f_offset=f_off,
            initial_phase=phi0,
            snr_db=100.0,    # effectively noiseless
            dc_offset_i=0.0,
            dc_offset_q=0.0,
            bits=bits.astype(np.int8),
        )
        clean_iq = gen.synthesize_clean(params)  # (2, 240) float32

        # ── RMS-match the clean target to the real window ─────────────────────
        # This corrects for any amplitude estimation error so the loss function
        # penalises shape differences, not just an overall scale mismatch.
        real_rms   = float(np.sqrt(np.mean(I_win ** 2 + Q_win ** 2))) + 1e-9
        clean_rms  = float(np.sqrt(np.mean(clean_iq ** 2))) + 1e-9
        clean_iq   = clean_iq * (real_rms / clean_rms)

        noisy_iq = np.stack([I_win, Q_win], axis=0)  # (2, 240)

        noisy_list.append(noisy_iq)
        clean_list.append(clean_iq)
        hex_list.append(hex_msg)
        pos_list.append(pos)
        amp_list.append(amp)
        foff_list.append(f_off)
        phi_list.append(phi0)

        prev_valid_pos = pos

    n_found = len(hex_list)
    if verbose:
        print(f"  {npy_path.name}: {n_found} valid frames extracted")

    stats = {
        "source_file": npy_path.name,
        "n_samples": n_samples,
        "candidates": int(len(candidates)),
        "valid_frames": n_found,
        "noise_floor": float(noise_floor),
        "threshold": float(thr),
    }

    if n_found == 0:
        return _empty_result(stats)

    return {
        "noisy_iq":     np.stack(noisy_list, axis=0),              # (N, 2, 240)
        "clean_iq":     np.stack(clean_list, axis=0),              # (N, 2, 240)
        "hex_msgs":     hex_list,
        "positions":    np.array(pos_list, dtype=np.int32),
        "amp_est":      np.array(amp_list,  dtype=np.float32),
        "f_offset_est": np.array(foff_list, dtype=np.float32),
        "phase_est":    np.array(phi_list,  dtype=np.float32),
        "stats":        stats,
    }


def _empty_result(stats: dict) -> Dict:
    return {
        "noisy_iq":     np.zeros((0, 2, FRAME_SAMPLES), dtype=np.float32),
        "clean_iq":     np.zeros((0, 2, FRAME_SAMPLES), dtype=np.float32),
        "hex_msgs":     [],
        "positions":    np.array([], dtype=np.int32),
        "amp_est":      np.array([], dtype=np.float32),
        "f_offset_est": np.array([], dtype=np.float32),
        "phase_est":    np.array([], dtype=np.float32),
        "stats":        stats,
    }


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract supervised training pairs (noisy real IQ ↔ clean re-synthesized IQ) "
            "from adsb_iq_sample_collection .npy burst files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   required=True,
                   help="Directory containing .npy IQ burst files.")
    p.add_argument("--labels-dir", required=True,
                   help="Output directory for *_labels.npz files.")
    p.add_argument("--recursive",  action="store_true",
                   help="Search --data-dir recursively for .npy files.")
    p.add_argument("--max-files",  type=int, default=None,
                   help="Process at most this many files (for quick tests).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip .npy files whose *_labels.npz already exists.")
    p.add_argument("--thr-percentile", type=float, default=90.0,
                   help="Noise-floor percentile for adaptive preamble threshold.")
    p.add_argument("--thr-mult",   type=float, default=3.5,
                   help="Noise-floor multiplier for preamble high threshold.")
    p.add_argument("--thr-min",    type=float, default=0.05,
                   help="Absolute minimum preamble threshold.")
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-file progress.")
    p.add_argument("--summary-json", type=str, default=None,
                   help="Optional path to write a JSON extraction summary.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    if not HAS_PYMODES:
        print("ERROR: pyModeS not found.  Install with:  pip install pyModeS",
              file=sys.stderr)
        sys.exit(1)

    args = _build_parser().parse_args(argv)

    data_dir   = Path(args.data_dir)
    labels_dir = Path(args.labels_dir)

    if not data_dir.exists():
        print(f"ERROR: --data-dir '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    labels_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.npy" if args.recursive else "*.npy"
    files = sorted(data_dir.glob(pattern))
    # Exclude any *_labels.npz mismatches (just in case)
    files = [f for f in files if not f.name.endswith("_labels.npy")]

    if not files:
        print(f"No .npy files found in {data_dir} (pattern: {pattern})")
        sys.exit(0)

    if args.max_files:
        files = files[: args.max_files]

    print(f"Found {len(files)} .npy file(s) to process.")
    print(f"Labels output dir: {labels_dir}\n")

    total_frames = 0
    skipped = 0
    all_stats = []
    t0_global = time.time()

    for i, npy_path in enumerate(files, 1):
        label_path = labels_dir / (npy_path.stem + "_labels.npz")

        if args.skip_existing and label_path.exists():
            skipped += 1
            if args.verbose:
                print(f"[{i:>4}/{len(files)}] SKIP  {npy_path.name}")
            continue

        t0 = time.time()
        result = extract_file(
            npy_path,
            preamble_thr_percentile=args.thr_percentile,
            preamble_thr_multiplier=args.thr_mult,
            preamble_thr_min=args.thr_min,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0

        n_found = len(result["hex_msgs"])
        total_frames += n_found

        # Save label file even if 0 frames (marks file as processed)
        np.savez_compressed(
            label_path,
            noisy_iq    = result["noisy_iq"],
            clean_iq    = result["clean_iq"],
            positions   = result["positions"],
            amp_est     = result["amp_est"],
            f_offset_est= result["f_offset_est"],
            phase_est   = result["phase_est"],
            # hex_msgs stored as object array for variable-length strings
            hex_msgs    = np.array(result["hex_msgs"], dtype=object),
        )
        all_stats.append(result["stats"])

        print(
            f"[{i:>4}/{len(files)}]  {npy_path.name:<45}  "
            f"{n_found:>5} frames  ({elapsed:.1f}s)"
        )

    elapsed_total = time.time() - t0_global
    print(f"\n{'─' * 60}")
    print(f"Processed : {len(files) - skipped} file(s)  |  skipped: {skipped}")
    print(f"Total valid frames extracted : {total_frames:,}")
    print(f"Total time                   : {elapsed_total:.1f}s")
    print(f"Labels saved to              : {labels_dir}")

    if args.summary_json:
        summary = {
            "total_files":  len(files),
            "skipped":      skipped,
            "total_frames": total_frames,
            "elapsed_s":    round(elapsed_total, 1),
            "per_file":     all_stats,
        }
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))
        print(f"Summary JSON → {args.summary_json}")


if __name__ == "__main__":
    main()
