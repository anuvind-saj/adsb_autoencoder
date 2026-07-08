"""
compare_decodings.py — Compare ADS-B decoding quality between two RTL-SDR binary files.

Decodes both a raw capture and a processed (autoencoder-cleaned) capture using
a pure-Python preamble detector + pyModeS CRC validation, then prints a
side-by-side comparison of valid frames and unique aircraft detected.

Typical usage
-------------
# Compare raw vs supervised-clean output:
python compare_decodings.py adsb_capture.bin adsb_capture_supervised_clean.bin

# Explicit labels:
python compare_decodings.py \\
    --raw   adsb_capture.bin \\
    --clean adsb_capture_supervised_clean.bin \\
    --label "supervised 50ep"

# Show the first N decoded hex messages for inspection:
python compare_decodings.py adsb_capture.bin adsb_capture_supervised_clean.bin --show 10

Input format
------------
Standard RTL-SDR interleaved uint8 binary: I0 Q0 I1 Q1 ...
This is the format written by live_bridge.py --output and rtl_sdr captures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# pyModeS CRC import
# ---------------------------------------------------------------------------

try:
    from pyModeS.util import crc as _pms_crc
    def _crc(hex_msg: str) -> int:
        return _pms_crc(hex_msg)
except Exception:
    print("ERROR: pyModeS not found.  Install with:  pip install pyModeS", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_SAMPLES: int = 240
PREAMBLE_HIGH_IDX: List[int] = [0, 2, 7, 9]
PREAMBLE_LOW_IDX:  List[int] = [1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

UINT8_CENTER: float = 127.5
UINT8_SCALE:  float = 127.5


def load_iq_normalized(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load interleaved uint8 IQ and return DC-removed float32 I, Q in [-1, 1]."""
    raw = np.fromfile(path, dtype=np.uint8)
    I = (raw[0::2].astype(np.float32) - UINT8_CENTER) / UINT8_SCALE
    Q = (raw[1::2].astype(np.float32) - UINT8_CENTER) / UINT8_SCALE
    I -= I.mean()
    Q -= Q.mean()
    return I, Q


def iq_normalized_to_bytes(I: np.ndarray, Q: np.ndarray) -> bytes:
    """Convert normalized float32 I/Q back to interleaved uint8 bytes."""
    out = np.empty(len(I) * 2, dtype=np.float32)
    out[0::2] = I
    out[1::2] = Q
    out = np.clip(out * UINT8_SCALE + UINT8_CENTER, 0.0, 255.0)
    return out.astype(np.uint8).tobytes()


def decode_iq(
    I: np.ndarray,
    Q: np.ndarray,
    thr_percentile: float = 90.0,
    thr_multiplier: float = 3.5,
    thr_min: float = 0.10,
    low_ratio: float = 3.0,
    show_n: int = 0,
    verbose: bool = True,
) -> Tuple[List[str], List[str], dict]:
    """
    Decode valid ADS-B Mode-S frames from normalized float32 I/Q arrays.

    Returns (messages, icao_list, stats).
    """
    mag = np.sqrt(I.astype(np.float32) ** 2 + Q.astype(np.float32) ** 2)

    noise_floor = float(np.percentile(mag, thr_percentile))
    thr = max(noise_floor * thr_multiplier, thr_min)
    if verbose:
        print(f"  noise_floor={noise_floor:.4f}  threshold={thr:.4f}  mag_max={mag.max():.3f}")

    if len(mag) < 16:
        if verbose:
            print("  ERROR: file too short.")
        return [], [], {"noise_floor": noise_floor, "threshold": thr, "candidates": 0}

    views = np.lib.stride_tricks.sliding_window_view(mag, 16)
    high = views[:, PREAMBLE_HIGH_IDX].mean(axis=1)
    low = views[:, PREAMBLE_LOW_IDX].mean(axis=1)
    candidates = np.where((high > thr) & (high > low * low_ratio))[0]
    if verbose:
        print(f"  Preamble candidates: {len(candidates):,}")

    messages: List[str] = []
    icao_set: set = set()
    prev_valid = -FRAME_SAMPLES
    shown = 0

    for pos in candidates:
        end = pos + FRAME_SAMPLES
        if end > len(mag):
            break
        if pos < prev_valid + FRAME_SAMPLES:
            continue

        data = mag[pos + 16: end].reshape(112, 2)
        bits = (data[:, 0] > data[:, 1]).astype(np.uint8)
        hex_msg = np.packbits(bits).tobytes().hex().upper()

        if _crc(hex_msg) == 0:
            messages.append(hex_msg)
            icao_set.add(hex_msg[2:8])
            prev_valid = pos

            if show_n > 0 and shown < show_n:
                print(f"    [{shown+1:>3}] pos={pos:>8}  {hex_msg[:28]}...  "
                      f"ICAO={hex_msg[2:8]}")
                shown += 1

    stats = {
        "noise_floor": noise_floor,
        "threshold": thr,
        "candidates": int(len(candidates)),
        "mag_max": float(mag.max()),
    }
    return messages, list(icao_set), stats


def decode_file(
    path: str | Path,
    thr_percentile: float = 90.0,
    thr_multiplier: float = 3.5,
    thr_min: float = 0.10,
    low_ratio: float = 3.0,
    show_n: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Decode all valid ADS-B Mode-S frames from an RTL-SDR binary file.

    Parameters
    ----------
    path            : Path to interleaved uint8 IQ binary file.
    thr_percentile  : Noise-floor percentile for adaptive preamble threshold.
    thr_multiplier  : Multiplier applied to noise floor for preamble high threshold.
    thr_min         : Absolute minimum threshold (prevents false positives on silence).
    low_ratio       : ``high_mean > low_mean * ratio`` preamble shape test.
    show_n          : Print the first N decoded hex messages (0 = print none).

    Returns
    -------
    messages : list of hex strings for each CRC-valid frame
    icao_set : list of unique ICAO addresses (bytes 2-4 of each message)
    """
    path = Path(path)
    print(f"\nLoading {path.name}  ({path.stat().st_size / 1e6:.1f} MB) ...")

    I, Q = load_iq_normalized(path)
    messages, icao_list, _ = decode_iq(
        I, Q,
        thr_percentile=thr_percentile,
        thr_multiplier=thr_multiplier,
        thr_min=thr_min,
        low_ratio=low_ratio,
        show_n=show_n,
        verbose=True,
    )
    return messages, icao_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare ADS-B decoding quality between raw and autoencoder-processed IQ captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("raw",   nargs="?", help="Raw RTL-SDR binary file (uint8 IQ).")
    p.add_argument("clean", nargs="?", help="Autoencoder-processed binary file (uint8 IQ).")
    p.add_argument("--raw",   dest="raw_flag",   metavar="FILE",
                   help="Explicit path to raw file (overrides positional).")
    p.add_argument("--clean", dest="clean_flag", metavar="FILE",
                   help="Explicit path to clean file (overrides positional).")
    p.add_argument("--label", type=str, default="autoencoder",
                   help="Short label for the clean file shown in the summary.")
    p.add_argument("--show",  type=int, default=0, metavar="N",
                   help="Print the first N decoded hex messages per file.")
    p.add_argument("--thr-percentile", type=float, default=90.0)
    p.add_argument("--thr-mult",       type=float, default=3.5)
    p.add_argument("--thr-min",        type=float, default=0.10)
    p.add_argument("--low-ratio",      type=float, default=3.0)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    raw_path   = args.raw_flag   or args.raw
    clean_path = args.clean_flag or args.clean

    if not raw_path or not clean_path:
        print("Usage:  python compare_decodings.py <raw.bin> <clean.bin>", file=sys.stderr)
        sys.exit(1)

    for p in (raw_path, clean_path):
        if not Path(p).exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    decoder_kw = dict(
        thr_percentile=args.thr_percentile,
        thr_multiplier=args.thr_mult,
        thr_min=args.thr_min,
        low_ratio=args.low_ratio,
        show_n=args.show,
    )

    raw_msgs,   raw_icao   = decode_file(raw_path,   **decoder_kw)
    clean_msgs, clean_icao = decode_file(clean_path, **decoder_kw)

    r, ri = len(raw_msgs),   len(raw_icao)
    c, ci = len(clean_msgs), len(clean_icao)
    df    = c - r
    di    = ci - ri

    label = args.label
    bar   = "=" * 50
    print(f"\n{bar}")
    print(f"  {'File':<22}  {'Frames':>7}  {'Aircraft':>8}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*8}")
    print(f"  {'Raw':<22}  {r:>7,}  {ri:>8,}")
    print(f"  {label:<22}  {c:>7,}  {ci:>8,}")
    print(f"  {'Delta':<22}  {df:>+7,}  {di:>+8,}")
    pct_f = 100.0 * df / max(r, 1)
    pct_i = 100.0 * di / max(ri, 1)
    print(f"  {'Change':<22}  {pct_f:>+6.1f}%  {pct_i:>+7.1f}%")
    print(f"{bar}")

    if c == 0 and r > 0:
        print("\nWARNING: clean file decoded 0 frames.")
        print("  Possible causes:")
        print("  • Model not yet fine-tuned on real data (domain shift)")
        print("  • live_bridge.py used a different checkpoint (check --ckpt)")
        print("  • Noise floor raised by the model — try lowering --thr-mult")


if __name__ == "__main__":
    main()
