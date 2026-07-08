"""
real_dataset.py — PyTorch Dataset for real-world ADS-B IQ captures.

Data format produced by adsb_iq_sample_collection/collector.py
==============================================================
* File extension : .npy
* dtype          : np.complex64
* shape          : (SAMPLES_PER_BURST,)  →  10 000 000 samples (5 s @ 2 MSPS)
* I channel      : array.real  — normalized to [-1, 1]  (uint8 - 127.5) / 127.5
* Q channel      : array.imag  — same normalization
* DC offset      : NOT removed at capture time; raw RTL-SDR hardware bias is present
* Sidecar JSON   : antenna_id, center_frequency_hz, sample_rate_sps, gain_db,
                   num_samples, start_time_utc, filename_iq

Self-supervised denoising strategy
-----------------------------------
No "perfectly clean" ADS-B reference exists for real captures, so we use a
Noise2Signal approach:

    1. Remove the per-file DC bias (mean-subtract I and Q independently).
    2. Treat the DC-cleaned real signal as the "clean target".
    3. Add extra Gaussian noise at a uniformly-sampled SNR (default 5–20 dB)
       to form the "noisy input".

The model learns to recover the real signal distribution from noisier versions,
which transfers to suppressing the original hardware noise at inference time.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["RealIQDataset", "WINDOW_SIZE"]

WINDOW_SIZE: int = 240  # must match IQAutoencoder's seq_len


class RealIQDataset(Dataset):
    """
    Dataset wrapping real-world IQ bursts from adsb_iq_sample_collection.

    Parameters
    ----------
    data_dir : str | Path
        Root directory to search for ``*.npy`` burst files.  Sub-directories
        are searched recursively when ``recursive=True``.
    window_size : int
        Samples per model input frame.  Must match ``IQAutoencoder`` seq_len
        (default 240 = one ADS-B Mode-S frame at 2 MSPS).
    hop : int
        Step between consecutive windows.  ``hop = window_size`` gives
        non-overlapping windows; ``hop = window_size // 2`` doubles the number
        of training samples at the cost of correlation between neighbors.
        Default: 120 (50 % overlap).
    noise_snr_db_range : tuple[float, float]
        Uniform SNR range ``[lo, hi]`` in dB for the added noise augmentation.
        Lower bound ≈ 5 dB models heavily corrupted signals; upper bound ≈ 20 dB
        gives mild noise.  The model learns to cover the whole range.
    max_files : int | None
        Cap the number of .npy files loaded (useful for quick experiments).
    file_glob : str
        Glob pattern relative to ``data_dir``.  Change to
        ``"**/*.npy"`` together with ``recursive=True`` to scan sub-directories.
    recursive : bool
        If True, scan ``data_dir`` recursively for ``*.npy`` files.
    seed : int
        RNG seed for reproducible noise augmentation.
    dc_remove : bool
        Subtract per-file mean I/Q before windowing.  Should always be True
        for RTL-SDR data to remove the hardware DC spike.
    cache_size : int
        Number of most-recently-used .npy files to keep in RAM.  Each file
        is ~80 MB (complex64 @ 10 M samples).  Default 4 → ≈ 320 MB.
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = WINDOW_SIZE,
        hop: int = 120,
        noise_snr_db_range: Tuple[float, float] = (5.0, 20.0),
        max_files: Optional[int] = None,
        file_glob: str = "*.npy",
        recursive: bool = False,
        seed: int = 42,
        dc_remove: bool = True,
        cache_size: int = 4,
    ) -> None:
        super().__init__()

        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if hop < 1:
            raise ValueError(f"hop must be >= 1, got {hop}")
        snr_lo, snr_hi = noise_snr_db_range
        if snr_lo > snr_hi:
            raise ValueError(f"noise_snr_db_range lo must be <= hi, got {noise_snr_db_range}")

        self.window_size = window_size
        self.hop = hop
        self.snr_lo = float(snr_lo)
        self.snr_hi = float(snr_hi)
        self.dc_remove = dc_remove
        self.cache_size = max(1, cache_size)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()  # guard LRU cache for DataLoader workers

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        pattern = f"**/{file_glob}" if recursive else file_glob
        files: List[Path] = sorted(data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {data_dir}. "
                "Copy .npy bursts from the Pi first."
            )

        if max_files is not None:
            files = files[:max_files]

        # ── Build flat window index: list of (file_path, start_sample) ────────
        self._index: List[Tuple[Path, int]] = []
        self._file_window_counts: Dict[Path, int] = {}

        for f in files:
            try:
                arr = np.load(f, mmap_mode="r")
            except Exception as exc:
                import warnings
                warnings.warn(f"Skipping unreadable file {f}: {exc}")
                continue

            n = arr.shape[0]
            if n < window_size:
                continue

            count = (n - window_size) // hop + 1
            self._file_window_counts[f] = count
            for s in range(0, count * hop, hop):
                self._index.append((f, s))

        if not self._index:
            raise RuntimeError(
                "No usable windows found. "
                f"window_size={window_size}, hop={hop}."
            )

        # ── LRU cache: {path: (I_array, Q_array)} ──────────────────────────────
        self._cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_order: List[Path] = []  # LRU order (oldest first)

        total_files = len(self._file_window_counts)
        total_windows = len(self._index)
        print(
            f"[RealIQDataset] {total_files} files  |  "
            f"{total_windows:,} windows  "
            f"(window={window_size}, hop={hop})"
        )

    # ── public properties ──────────────────────────────────────────────────────

    @property
    def num_files(self) -> int:
        return len(self._file_window_counts)

    def __len__(self) -> int:
        return len(self._index)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a .npy burst, extract I/Q, optionally remove DC, cache it."""
        with self._lock:
            if path in self._cache:
                # Move to most-recently-used position
                self._cache_order.remove(path)
                self._cache_order.append(path)
                return self._cache[path]

            arr = np.load(path, mmap_mode="r")
            I = arr.real.astype(np.float32)
            Q = arr.imag.astype(np.float32)

            if self.dc_remove:
                I = I - float(I.mean())
                Q = Q - float(Q.mean())

            # Evict oldest entry if cache is full
            if len(self._cache_order) >= self.cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]

            self._cache[path] = (I, Q)
            self._cache_order.append(path)
            return I, Q

    def _add_noise(
        self, clean: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return ``clean + Gaussian noise`` at a random SNR in ``[snr_lo, snr_hi]`` dB."""
        sig_power = float(np.mean(clean ** 2)) + 1e-9
        snr_db = rng.uniform(self.snr_lo, self.snr_hi)
        noise_power = sig_power * 10.0 ** (-snr_db / 10.0)
        noise = rng.standard_normal(clean.shape).astype(np.float32)
        return clean + noise * float(np.sqrt(noise_power))

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        noisy_input  : float32 tensor  shape (2, window_size)
        clean_target : float32 tensor  shape (2, window_size)
        """
        path, start = self._index[idx]
        end = start + self.window_size

        I_arr, Q_arr = self._load_file(path)
        I_win = I_arr[start:end]
        Q_win = Q_arr[start:end]

        clean = np.stack([I_win, Q_win], axis=0)  # (2, 240), float32

        # Use a local RNG derived from idx for reproducibility even with
        # multiple DataLoader workers (each worker gets a forked copy of
        # self._rng which would diverge; idx-based seeding avoids that).
        local_rng = np.random.default_rng(int(self._rng.integers(2**31)) ^ idx)
        noisy = self._add_noise(clean, local_rng)

        return torch.from_numpy(noisy), torch.from_numpy(clean)

    # ── utility ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        n_files = self.num_files
        n_windows = len(self)
        duration_per_file_s = (
            (self._index[-1][1] + self.window_size) / 2_000_000
            if self._index
            else 0
        )
        return (
            f"RealIQDataset:\n"
            f"  files        : {n_files}\n"
            f"  windows      : {n_windows:,}\n"
            f"  window_size  : {self.window_size}\n"
            f"  hop          : {self.hop}\n"
            f"  SNR range    : [{self.snr_lo}, {self.snr_hi}] dB\n"
            f"  dc_remove    : {self.dc_remove}\n"
            f"  cache_size   : {self.cache_size} files\n"
        )
