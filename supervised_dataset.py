"""
supervised_dataset.py — PyTorch Dataset for supervised real-world training pairs.

Loads the *_labels.npz files produced by extract_labels.py and returns
(noisy_real_iq, clean_synthesized_iq) pairs ready for model training.

Each pair is:
  noisy_input  : float32 (2, 240) — real RTL-SDR captured IQ window
                 (DC removed, amplitude-normalised)
  clean_target : float32 (2, 240) — re-synthesized perfect IQ using the
                 decoded 112 bits, aligned in amplitude/phase/frequency
                 to the real window

Why this is better than self-supervised (real_dataset.py)
----------------------------------------------------------
The clean target here is a mathematically perfect ADS-B signal, not just
the real hardware signal with extra noise added.  The model learns to map
real noisy captures → perfect signals, which directly improves CRC-passing
rates at inference time.

Optional augmentation
---------------------
Three lightweight augmentations are supported:

  amplitude_jitter : Multiply both noisy and clean by a uniform random
                     scale in [1-jitter, 1+jitter].  Teaches scale invariance.

  phase_rotation   : Rotate the complex plane of BOTH tensors by the same
                     random angle.  Teaches phase invariance.  This is safe
                     because the clean target is aligned to the noisy input.

  collision_augment: (fraction in [0, 1]) — On each __getitem__ call,
                     with this probability a synthetic second-aircraft
                     signal is linearly added on top of the noisy input.
                     The clean target remains the original single-aircraft
                     clean signal.  This teaches the model to separate
                     co-channel collisions using data whose noise floor is
                     real hardware noise (not fully synthetic).

                     The interferer amplitude is drawn from
                     Uniform(amp_ratio_min, amp_ratio_max) × signal_A_amp,
                     where signal_A_amp is the per-frame estimate stored by
                     extract_labels.py in the amp_est field.

                     The interferer frequency offset is drawn at least
                     collision_min_freq_sep_hz away from signal A's
                     estimated offset, ensuring a visible beating envelope
                     within the 240-sample window.

                     A random 112-bit payload and random time offset
                     (collision_time_offset_range samples) are used, so
                     each training call produces a unique collision geometry.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# generator.py lives in the same package directory
from generator import ADSBSignalGenerator, SignalParams, FRAME_SAMPLES

__all__ = ["SupervisedIQDataset"]


# ---------------------------------------------------------------------------
# Module-level generator (one per process — re-entrant via per-call RNG)
# ---------------------------------------------------------------------------
_GEN = ADSBSignalGenerator()


class SupervisedIQDataset(Dataset):
    """
    Dataset wrapping *_labels.npz files from extract_labels.py.

    Parameters
    ----------
    labels_dir      : Directory containing *_labels.npz files.
    file_glob       : Glob pattern for label files.
    recursive       : Search labels_dir recursively.
    max_files       : Cap the number of .npz files loaded.
    amplitude_jitter: Float in [0, 1).  If > 0, both noisy and clean tensors
                      are multiplied by Uniform(1-jitter, 1+jitter) per sample.
                      Prevents the model from relying on absolute amplitude.
    phase_rotation  : If True, both tensors are rotated by a random angle
                      in [0, 2π) on the complex plane per sample.
    normalise       : If True, divide both tensors by the noisy window RMS so
                      the model input is always unit-power.  The clean target is
                      scaled by the same factor, preserving their relative scale.
    collision_augment: Probability ∈ [0, 1] that each sample has a synthetic
                      second-aircraft signal added.  0.0 = disabled.
                      Recommended: 0.10–0.15 so the model sees ~1 collision
                      per 7–10 clean examples.  Higher values (≥0.30) cause
                      over-suppression of primary signal structure at inference.
    amp_ratio_min   : Minimum interferer amplitude as a fraction of signal A.
                      Default 0.3 (−10 dB weak interferer).
    amp_ratio_max   : Maximum interferer amplitude as a fraction of signal A.
                      Default 1.5 (+3.5 dB dominant interferer).
    collision_min_freq_sep_hz: Minimum |f_a − f_b| to guarantee a visible
                      beating cycle within 240 samples.  Default 8 000 Hz.
    collision_time_offset_range: (min, max) sample offset of signal B relative
                      to signal A.  Default (−24, 24) — up to ±12 µs.
    f_offset_range  : Full carrier frequency range (Hz) used to draw the
                      interferer offset when signal A's own offset is unknown.
                      Default (−50 000, 50 000) Hz.
    interferer_snr_db: SNR (dB) of AWGN added to the synthetic interferer
                      before superimposing.  Adding noise prevents the model
                      from learning to suppress all perfectly-clean structured
                      patterns (which caused over-suppression in v2).
                      Default 20.0 dB (light noise, realistic for a mid-range
                      aircraft).  Set to None to disable.
    seed            : RNG seed for reproducible augmentation.
    """

    def __init__(
        self,
        labels_dir: str | Path,
        file_glob: str = "*_labels.npz",
        recursive: bool = False,
        max_files: Optional[int] = None,
        amplitude_jitter: float = 0.1,
        phase_rotation: bool = True,
        normalise: bool = True,
        collision_augment: float = 0.0,
        amp_ratio_min: float = 0.3,
        amp_ratio_max: float = 1.5,
        collision_min_freq_sep_hz: float = 8_000.0,
        collision_time_offset_range: Tuple[int, int] = (-24, 24),
        f_offset_range: Tuple[float, float] = (-50_000.0, 50_000.0),
        interferer_snr_db: Optional[float] = 20.0,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.amplitude_jitter   = float(amplitude_jitter)
        self.phase_rotation     = bool(phase_rotation)
        self.normalise          = bool(normalise)
        self.collision_augment  = float(np.clip(collision_augment, 0.0, 1.0))
        self.amp_ratio_min      = float(amp_ratio_min)
        self.amp_ratio_max      = float(amp_ratio_max)
        self.collision_min_freq_sep_hz  = float(collision_min_freq_sep_hz)
        self.collision_time_offset_range = tuple(collision_time_offset_range)
        self.f_offset_range     = tuple(f_offset_range)
        self.interferer_snr_db  = interferer_snr_db  # None = no noise on interferer
        self._rng  = np.random.default_rng(seed)
        self._lock = threading.Lock()

        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            raise FileNotFoundError(
                f"labels_dir not found: {labels_dir}\n"
                "Run extract_labels.py first to generate label files."
            )

        pattern = f"**/{file_glob}" if recursive else file_glob
        npz_files = sorted(labels_dir.glob(pattern))
        if max_files is not None:
            npz_files = npz_files[:max_files]

        if not npz_files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {labels_dir}.\n"
                "Run extract_labels.py first."
            )

        # ── Load all pairs into RAM (label files are small) ──────────────────
        noisy_chunks:    List[np.ndarray] = []
        clean_chunks:    List[np.ndarray] = []
        amp_est_chunks:  List[np.ndarray] = []
        foff_est_chunks: List[np.ndarray] = []
        self._hex_msgs:    List[str] = []
        self._source_files: List[str] = []

        total_frames = 0
        skipped_empty = 0

        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            noisy = data["noisy_iq"]   # (N, 2, 240)
            clean = data["clean_iq"]   # (N, 2, 240)

            if noisy.shape[0] == 0:
                skipped_empty += 1
                continue

            N = noisy.shape[0]
            noisy_chunks.append(noisy.astype(np.float32))
            clean_chunks.append(clean.astype(np.float32))

            # Per-frame amplitude and frequency offset estimates saved by
            # extract_labels.py — used to calibrate the synthetic interferer.
            amp_est  = data.get("amp_est",  None)
            foff_est = data.get("f_offset_est", None)
            amp_est_chunks.append(
                amp_est.astype(np.float32) if amp_est is not None
                else np.ones(N, dtype=np.float32)
            )
            foff_est_chunks.append(
                foff_est.astype(np.float32) if foff_est is not None
                else np.zeros(N, dtype=np.float32)
            )

            hex_arr = data.get("hex_msgs", np.array([], dtype=object))
            self._hex_msgs.extend(hex_arr.tolist())
            self._source_files.extend([npz_path.stem] * N)
            total_frames += N

        if total_frames == 0:
            raise RuntimeError(
                "All label files are empty (0 valid frames found).\n"
                "Try loosening the preamble threshold in extract_labels.py or "
                "check that your .npy files are valid RTL-SDR captures."
            )

        self._noisy    = np.concatenate(noisy_chunks,    axis=0)  # (Total, 2, 240)
        self._clean    = np.concatenate(clean_chunks,    axis=0)  # (Total, 2, 240)
        self._amp_est  = np.concatenate(amp_est_chunks,  axis=0)  # (Total,)
        self._foff_est = np.concatenate(foff_est_chunks, axis=0)  # (Total,)

        aug_desc_parts = []
        if amplitude_jitter > 0 or phase_rotation:
            aug_desc_parts.append("amplitude/phase jitter")
        if self.collision_augment > 0:
            aug_desc_parts.append(
                f"collision_augment={self.collision_augment:.0%}"
            )
        aug_desc = " + ".join(aug_desc_parts) if aug_desc_parts else "no augmentation"

        n_files_loaded = len(npz_files) - skipped_empty
        print(
            f"[SupervisedIQDataset] {n_files_loaded} label file(s)  |  "
            f"{total_frames:,} training pairs  ({aug_desc})"
        )

    def __len__(self) -> int:
        return self._noisy.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy = self._noisy[idx].copy()   # (2, 240)
        clean = self._clean[idx].copy()   # (2, 240)

        # Thread-safe per-sample RNG derived from the shared seed and index
        with self._lock:
            rng_seed = int(self._rng.integers(2 ** 31)) ^ idx
        local_rng = np.random.default_rng(rng_seed)

        # ── Collision augmentation ────────────────────────────────────────────
        # With probability `collision_augment`, synthesise a second aircraft
        # signal and add it to the noisy input.  The clean target stays as
        # signal A only — the model learns to separate A from the composite.
        if self.collision_augment > 0.0 and local_rng.random() < self.collision_augment:
            noisy = self._add_synthetic_interferer(idx, noisy, local_rng)

        # ── Optional: normalise to unit RMS power ─────────────────────────────
        if self.normalise:
            rms = float(np.sqrt(np.mean(noisy ** 2))) + 1e-9
            noisy = noisy / rms
            clean = clean / rms

        # ── Augmentations (applied identically to both tensors) ───────────────
        if self.phase_rotation:
            angle   = local_rng.uniform(0.0, 2.0 * np.pi)
            cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            for arr in (noisy, clean):
                I_rot = arr[0] * cos_a - arr[1] * sin_a
                Q_rot = arr[0] * sin_a + arr[1] * cos_a
                arr[0], arr[1] = I_rot, Q_rot

        if self.amplitude_jitter > 0.0:
            scale = float(local_rng.uniform(
                1.0 - self.amplitude_jitter,
                1.0 + self.amplitude_jitter,
            ))
            noisy *= scale
            clean *= scale

        return torch.from_numpy(noisy), torch.from_numpy(clean)

    # ── Collision augmentation helper ─────────────────────────────────────────

    def _add_synthetic_interferer(
        self,
        idx: int,
        noisy_a: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Add a synthetic second-aircraft signal (signal B) to noisy_a.

        Signal B is generated from scratch using generator.py with parameters
        calibrated relative to signal A's estimated amplitude and frequency
        offset (stored in the label file by extract_labels.py).

        The returned array has shape (2, 240) and represents:
            I_composite = I_real_A + I_synthetic_B
            Q_composite = Q_real_A + Q_synthetic_B

        The clean target (signal A only) is unchanged — the model trains to
        project this composite back onto the clean single-aircraft signal.

        Calibration strategy
        --------------------
        Amplitude B  : Uniform(amp_ratio_min, amp_ratio_max) × amp_est[idx]
                       This ensures B is physically plausible relative to the
                       real received power of signal A.

        Frequency B  : Drawn from f_offset_range but forced to be at least
                       collision_min_freq_sep_hz away from f_offset_est[idx],
                       guaranteeing a visible beating cycle within 240 samples.

        Phase B      : Uniform [0, 2π) — unknown initial phase.
        Bits B       : Random 112-bit payload (collision partner identity
                       is irrelevant — only its signal geometry matters).
        Time offset  : Uniform integer in collision_time_offset_range —
                       covers partial preamble overlap and fully synchronised
                       collisions.
        """
        amp_a  = float(self._amp_est[idx])
        foff_a = float(self._foff_est[idx])

        # ── Amplitude of interferer ───────────────────────────────────────────
        amp_b = float(rng.uniform(self.amp_ratio_min, self.amp_ratio_max)) * amp_a
        amp_b = max(amp_b, 1e-4)   # floor to avoid degenerate zero

        # ── Frequency offset of interferer: must differ from A by min sep ─────
        f_lo, f_hi = self.f_offset_range
        sep = self.collision_min_freq_sep_hz

        # Draw f_b from the valid range excluding [foff_a - sep, foff_a + sep].
        # Split the allowed band into up to two sub-intervals, pick one
        # proportionally by width, then draw uniformly within it.
        low_band  = (f_lo, foff_a - sep)
        high_band = (foff_a + sep, f_hi)
        w_low  = max(0.0, low_band[1]  - low_band[0])
        w_high = max(0.0, high_band[1] - high_band[0])
        total_w = w_low + w_high

        if total_w < 1.0:
            # Edge case: f_offset_range too narrow — fall back to fixed sep
            foff_b = foff_a + sep * (1.0 if rng.random() > 0.5 else -1.0)
        else:
            if rng.random() < w_low / total_w:
                foff_b = float(rng.uniform(*low_band))
            else:
                foff_b = float(rng.uniform(*high_band))

        # ── Random payload, phase, and time offset ────────────────────────────
        bits_b = rng.integers(0, 2, size=112).astype(np.int8)
        phase_b = float(rng.uniform(0.0, 2.0 * np.pi))
        time_off = int(rng.integers(
            self.collision_time_offset_range[0],
            self.collision_time_offset_range[1] + 1,
        ))

        params_b = SignalParams(
            amplitude=amp_b,
            f_offset=foff_b,
            initial_phase=phase_b,
            bits=bits_b,
        )

        # ── Synthesise B and optionally add receiver AWGN ────────────────────
        # Adding noise to signal B is critical: a perfectly-clean synthetic
        # interferer causes the model to learn "suppress all clean structure",
        # which over-generalises to suppressing the primary signal at inference.
        # Injecting noise at a realistic SNR makes B indistinguishable from a
        # real distant aircraft, keeping the suppression behaviour targeted.
        clean_b = _GEN.synthesize_clean(params_b)   # (2, FRAME_SAMPLES)

        if self.interferer_snr_db is not None:
            sig_power = float(np.mean(clean_b ** 2)) + 1e-12
            noise_var = sig_power / (10.0 ** (self.interferer_snr_db / 10.0))
            clean_b = clean_b + rng.normal(
                0.0, float(np.sqrt(noise_var)), size=clean_b.shape
            ).astype(np.float32)

        # ── Apply time offset by rolling signal B ────────────────────────────
        if time_off != 0:
            shifted_b = np.zeros_like(clean_b)
            if time_off > 0:
                shifted_b[:, time_off:] = clean_b[:, : FRAME_SAMPLES - time_off]
            else:
                shifted_b[:, : FRAME_SAMPLES + time_off] = clean_b[:, -time_off:]
            clean_b = shifted_b

        # ── Linear field superposition (voltage addition, not power) ──────────
        return (noisy_a + clean_b).astype(np.float32)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def hex_msgs(self) -> List[str]:
        """Decoded hex messages corresponding to each training pair."""
        return self._hex_msgs

    def unique_icao_count(self) -> int:
        """Number of unique ICAO addresses (bytes 2–8 of each hex message)."""
        return len({m[2:8] for m in self._hex_msgs if len(m) >= 8})

    def summary(self) -> str:
        if self.collision_augment > 0:
            snr_str = (
                f"  interferer_snr: {self.interferer_snr_db:.0f} dB\n"
                if self.interferer_snr_db is not None else
                "  interferer_snr: clean (no noise)\n"
            )
            collision_line = (
                f"  collision_augment: {self.collision_augment:.0%}  "
                f"(amp_ratio [{self.amp_ratio_min:.1f}×, {self.amp_ratio_max:.1f}×]  "
                f"min_freq_sep {self.collision_min_freq_sep_hz/1e3:.0f} kHz  "
                f"time_off {self.collision_time_offset_range})\n"
                + snr_str
            )
        else:
            collision_line = "  collision_augment: disabled\n"
        return (
            f"SupervisedIQDataset:\n"
            f"  pairs            : {len(self):,}\n"
            f"  unique ICAO      : {self.unique_icao_count()}\n"
            f"  amplitude_jitter : {self.amplitude_jitter}\n"
            f"  phase_rotation   : {self.phase_rotation}\n"
            f"  normalise        : {self.normalise}\n"
            + collision_line
        )
