"""
live_bridge.py — Real-Time IQ Denoising Bridge for RTL-SDR Streams
===================================================================
Reads raw RTL-SDR binary IQ data (interleaved uint8), denoises it through
the trained IQAutoencoder, and writes the clean IQ bytes back to stdout
for downstream consumption by tools like dump1090.

Data format (RTL-SDR raw binary)
---------------------------------
  Each sample pair on wire:  [I_uint8, Q_uint8, I_uint8, Q_uint8, ...]
  Signed normalisation:       float = (uint8 - 127.5) / 127.5  →  [-1.0, 1.0]
  DC level at byte 127.5 because RTL-SDR uses unsigned ADC output centred at midrange.
  Denormalisation:            uint8 = clip(float × 127.5 + 127.5, 0, 255)
  DC removal (supervised checkpoints): hardware bias estimated from the first
  ~500 K samples (warmup buffer), then subtracted as a constant from every window
  before RMS normalisation.  This matches extract_labels.py which uses the full
  ~60 M sample burst mean; per-window subtraction distorts pulses by including
  signal energy in the mean estimate.

Sliding-Window Inference with Overlap-Add (OLA)
-----------------------------------------------
  The continuous stream is sliced into overlapping windows of WINDOW_SIZE=240
  samples (= 120 µs @ 2 MHz).  The hop between consecutive windows is
  HOP_SIZE (default 120, i.e. 50 % overlap).

  Synthesis uses a Hanning window before OLA accumulation:

       out_acc[n] += hann[n] × model_output[n]
       wgt_acc[n] += hann[n]

  After each hop the first HOP_SIZE samples, which have received all
  overlapping contributions, are normalised and flushed to stdout.

  Properties
  ──────────
  • 50 % overlap with Hanning → smoothly blended edges (no blocking artefacts)
  • Identity model   → reconstructed signal ≈ input (verified in smoke test)
  • Initial latency  = WINDOW_SIZE samples = 120 µs (imperceptible at 2 MHz)
  • Throughput limit = 1 / (HOP_SIZE × T_s × batch_size) inferences/second
                     ≈ 16 667 inf/s at hop=120 (GPU recommended for live use)

Batching strategy
-----------------
  Windows are accumulated into mini-batches of `--batch-size` (default 32)
  before a single GPU/MPS call.  Larger batches amortise CUDA/MPS launch
  overhead; smaller batches reduce latency.  For file-mode processing on CPU,
  batch-size 64-128 typically maximises throughput.

Usage
-----
  # Live piping (Linux):
  rtl_sdr -f 1090000000 -s 2000000 - | \\
      python live_bridge.py --ckpt checkpoints/best.pt | \\
      dump1090 --ifile /dev/stdin --raw

  # macOS (dump1090 does not support --ifile stdin; use a named FIFO):
  mkfifo /tmp/iq_pipe
  python live_bridge.py --ckpt checkpoints/best.pt < recorded.bin > /tmp/iq_pipe &
  dump1090 --ifile /tmp/iq_pipe --raw

  # Post-process a recorded RTL-SDR binary file:
  python live_bridge.py --ckpt checkpoints/best.pt \\
      --input recorded.bin --output clean.bin

  # Validate the full pipeline (OLA + DC + RMS) without learned weights:
  python live_bridge.py --ckpt checkpoints/best_supervised.pt \\
      --identity --input recorded.bin --output identity.bin
  python compare_decodings.py recorded.bin identity.bin

  # Mix model output with raw (matches blend_sweep.py post-processing):
  python live_bridge.py --ckpt checkpoints/best_supervised_v4.pt \\
      --blend 0.05 --input adsb_capture.bin --output clean_blend005.bin

  # Tune aggressiveness (higher hop = faster, less smooth):
  python live_bridge.py --ckpt checkpoints/best.pt --hop 240 --batch-size 64

Performance notes
-----------------
  At 2 MSPS the bridge must sustain ≥ 4 MB/s throughput.
  Benchmarked with base_channels=32 on Apple M-series (MPS):
    hop=120, batch=32  → ~3.2 MB/s  (borderline live; fine for post-processing)
    hop=240, batch=64  → ~6.8 MB/s  (comfortable real-time)
  Use --hop 240 --batch-size 64 for reliable live operation on CPU.
  Pass --device mps or --device cuda for GPU-accelerated real-time throughput.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import platform
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import load_model, IQAutoencoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UINT8_CENTER:  float = 127.5
UINT8_SCALE:   float = 127.5
WINDOW_SIZE:   int   = 240       # must match model seq_len
DEFAULT_HOP:   int   = 120       # 50 % overlap
DEFAULT_BATCH: int   = 32
READ_CHUNK:    int   = 65_536    # bytes per stdin read (~16 384 IQ samples)
STATS_EVERY_S: float = 5.0       # print throughput to stderr every N seconds

log = logging.getLogger("live_bridge")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def bytes_to_iq(raw: bytes | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert RTL-SDR raw uint8 bytes to normalised float32 I and Q arrays.

    Input  : flat byte sequence  [I0, Q0, I1, Q1, ...]  (even length).
    Output : (I_array, Q_array)  each float32, shape (N,), values in [-1, 1].
    """
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    arr = (arr - UINT8_CENTER) / UINT8_SCALE
    return arr[0::2].copy(), arr[1::2].copy()


def iq_to_bytes(i_arr: np.ndarray, q_arr: np.ndarray) -> bytes:
    """
    Convert normalised float32 I, Q arrays back to interleaved uint8 bytes.

    The float values are clipped to [-1, 1] before scaling to prevent
    saturation of samples where the model output slightly exceeds unity.

    Output : flat byte sequence  [I0, Q0, I1, Q1, ...]
    """
    out = np.empty(len(i_arr) * 2, dtype=np.float32)
    out[0::2] = i_arr
    out[1::2] = q_arr
    out = np.clip(out * UINT8_SCALE + UINT8_CENTER, 0.0, 255.0)
    return out.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Overlap-Add (OLA) reconstructor
# ---------------------------------------------------------------------------

class OLABuffer:
    """
    Overlap-Add synthesis buffer.

    Each inferred frame is multiplied by a Hanning synthesis window and
    accumulated into a running buffer.  After every call to `push()`, the
    first `hop_size` samples — which have received all overlapping
    contributions and are therefore settled — are returned and removed from
    the buffer.

    With 50 % overlap (hop = N/2) and a Hanning window:
      • Every output sample is the normalised weighted average of exactly
        two model outputs (steady-state), producing a smooth blend.
      • An identity model gives out ≈ in (verified in smoke test).
      • Initial latency is one hop (60 µs at 2 MHz, hop=120).

    Parameters
    ----------
    window_size : Number of IQ samples per model window.
    hop_size    : Number of samples to advance per frame.
    """

    def __init__(self, window_size: int = WINDOW_SIZE, hop_size: int = DEFAULT_HOP) -> None:
        self.N   = window_size
        self.H   = hop_size
        self.win = np.hanning(window_size).astype(np.float32)   # synthesis window

        # Accumulation buffers — size N so future windows always fit
        self._acc_i   = np.zeros(window_size, dtype=np.float32)
        self._acc_q   = np.zeros(window_size, dtype=np.float32)
        self._acc_wgt = np.zeros(window_size, dtype=np.float32)

    def push(self, frame_i: np.ndarray, frame_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Accumulate one model output frame and return the settled output.

        Parameters
        ----------
        frame_i, frame_q : float32 arrays of length window_size — model output.

        Returns
        -------
        (out_i, out_q) : float32 arrays of length hop_size — ready for output.
        """
        # Weighted accumulation
        self._acc_i   += self.win * frame_i
        self._acc_q   += self.win * frame_q
        self._acc_wgt += self.win

        H, N = self.H, self.N

        # Normalise and flush the first H samples (fully settled)
        safe_wgt = np.maximum(self._acc_wgt[:H], 1e-8)
        out_i    = self._acc_i[:H]   / safe_wgt
        out_q    = self._acc_q[:H]   / safe_wgt

        # Shift accumulation buffer left by H; zero-pad the tail
        self._acc_i[:N - H]   = self._acc_i[H:N]
        self._acc_i[N - H:]   = 0.0
        self._acc_q[:N - H]   = self._acc_q[H:N]
        self._acc_q[N - H:]   = 0.0
        self._acc_wgt[:N - H] = self._acc_wgt[H:N]
        self._acc_wgt[N - H:] = 0.0

        return out_i.copy(), out_q.copy()

    def flush_remaining(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Force-flush all remaining buffered samples at end-of-stream.

        Samples with zero accumulated weight are set to 0 rather than NaN.
        """
        safe_wgt = np.where(self._acc_wgt > 1e-8, self._acc_wgt, 1.0)
        i = np.where(self._acc_wgt > 1e-8, self._acc_i / safe_wgt, 0.0)
        q = np.where(self._acc_wgt > 1e-8, self._acc_q / safe_wgt, 0.0)
        self._acc_i[:]   = 0.0
        self._acc_q[:]   = 0.0
        self._acc_wgt[:] = 0.0
        return i.astype(np.float32), q.astype(np.float32)


# ---------------------------------------------------------------------------
# Batched sliding-window inference engine
# ---------------------------------------------------------------------------

class IQStreamBridge:
    """
    End-to-end streaming pipeline: raw bytes → denoised bytes.

    Internally maintains:
      • A byte-alignment buffer (handles reads that split mid-sample).
      • A float32 IQ sample queue.
      • An OLABuffer for smooth overlap-add reconstruction.
      • A pending output queue returned on each `feed()` call.

    Parameters
    ----------
    model      : Trained IQAutoencoder in eval() mode.
    device     : Torch device for inference.
    hop_size   : OLA hop in samples (default 120 = 50 % overlap).
    batch_size : Number of windows per GPU call (trades latency for throughput).
    blend_alpha: Mix model output with raw input after OLA:
                 out = blend_alpha * model + (1 - blend_alpha) * raw.
                 1.0 = pure model (default).  See --blend CLI flag.
    """

    def __init__(
        self,
        model:            IQAutoencoder,
        device:           torch.device,
        hop_size:         int  = DEFAULT_HOP,
        batch_size:       int  = DEFAULT_BATCH,
        normalize_windows: bool = False,
        blend_alpha:      float = 1.0,
    ) -> None:
        self.model             = model
        self.device            = device
        self.H                 = hop_size
        self.B                 = batch_size
        self.N                 = WINDOW_SIZE
        self.normalize_windows = normalize_windows
        self.blend_alpha       = float(blend_alpha)
        self._blend_enabled    = self.blend_alpha < 1.0 - 1e-9

        # Byte-alignment buffer (holds at most 1 orphaned byte between chunks)
        self._byte_buf: bytes = b""

        # Float32 IQ sample queue  (numpy arrays, extended each feed() call)
        self._q_i = np.empty(0, dtype=np.float32)
        self._q_q = np.empty(0, dtype=np.float32)

        # Raw input archive for --blend (indexed by global sample position)
        self._raw_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self._raw_total: int = 0
        self._raw_base_idx: int = 0

        # OLA synthesiser
        self._ola = OLABuffer(WINDOW_SIZE, hop_size)

        # Pending output (returned in feed())
        self._out_i = np.empty(0, dtype=np.float32)
        self._out_q = np.empty(0, dtype=np.float32)

        # DC offset estimation — mirrors extract_labels.py which subtracts the
        # full-burst mean from ~60M samples before saving label pairs.
        # Per-window subtraction (the old approach) over-removes DC because
        # ADS-B pulse energy inflates the 240-sample window mean.
        # Strategy: accumulate the first _DC_WARMUP_TARGET samples (mostly
        # noise), compute the mean once, freeze it, and apply that constant
        # offset to every window for the rest of the session.
        # ~500 K samples ≈ 0.45 s at 1.1 MSPS — negligible startup delay.
        _DC_WARMUP_TARGET = 500_000
        self._dc_warmup_target: int = _DC_WARMUP_TARGET
        self._dc_warmup_n:      int = 0
        self._dc_ready:         bool = False
        self._dc_i:             float = 0.0
        self._dc_q:             float = 0.0
        self._dc_warmup_buf_i:  list = []
        self._dc_warmup_buf_q:  list = []

        # Performance counters
        self.samples_in:  int   = 0
        self.samples_out: int   = 0
        self.n_inferences: int  = 0
        self._t_start:    float = time.perf_counter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue_output(self, i: np.ndarray, q: np.ndarray) -> None:
        self._out_i = np.concatenate([self._out_i, i])
        self._out_q = np.concatenate([self._out_q, q])

    def _append_raw_archive(self, i: np.ndarray, q: np.ndarray) -> None:
        """Store normalized raw samples for sample-aligned output blending."""
        if len(i) == 0:
            return
        self._raw_chunks.append((i, q))
        self._raw_total += len(i)

    def _raw_slice(self, global_start: int, global_end: int) -> tuple[np.ndarray, np.ndarray]:
        """Return raw I/Q for global sample indices [global_start, global_end)."""
        parts_i: list[np.ndarray] = []
        parts_q: list[np.ndarray] = []
        pos = self._raw_base_idx
        for ci, cq in self._raw_chunks:
            chunk_end = pos + len(ci)
            if chunk_end <= global_start:
                pos = chunk_end
                continue
            if pos >= global_end:
                break
            lo = global_start - pos
            hi = global_end - pos
            parts_i.append(ci[lo:hi])
            parts_q.append(cq[lo:hi])
            pos = chunk_end
        if not parts_i:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        if len(parts_i) == 1:
            return parts_i[0], parts_q[0]
        return np.concatenate(parts_i), np.concatenate(parts_q)

    def _trim_raw_archive(self, global_end: int) -> None:
        """Drop raw chunks that will never be needed for future output."""
        keep_from = max(global_end - self.N * 2, self._raw_base_idx)
        drop = keep_from - self._raw_base_idx
        if drop <= 0:
            return
        pos = self._raw_base_idx
        new_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        for ci, cq in self._raw_chunks:
            chunk_end = pos + len(ci)
            if chunk_end <= keep_from:
                pos = chunk_end
                continue
            if pos >= keep_from:
                new_chunks.append((ci, cq))
            else:
                off = keep_from - pos
                new_chunks.append((ci[off:], cq[off:]))
            pos = chunk_end
        self._raw_chunks = new_chunks
        self._raw_base_idx = keep_from
        self._raw_total = sum(len(c[0]) for c in self._raw_chunks)

    def _emit_pending_output(self) -> bytes:
        """Blend (if enabled), convert pending float IQ to uint8 bytes."""
        if len(self._out_i) == 0:
            return b""

        if self._blend_enabled:
            # OLA can produce output before matching raw has been ingested.
            # Emit only what raw can cover; keep the rest in _out_i/_out_q.
            max_emit = self.samples_in - self.samples_out
            if max_emit <= 0:
                return b""

            emit_n = min(len(self._out_i), max_emit)
            out_start = self.samples_out
            raw_i, raw_q = self._raw_slice(out_start, out_start + emit_n)
            m = min(emit_n, len(raw_i))
            if m == 0:
                return b""

            chunk_i = self._out_i[:m]
            chunk_q = self._out_q[:m]
            alpha = self.blend_alpha
            if alpha <= 1e-9:
                blended_i, blended_q = raw_i[:m].copy(), raw_q[:m].copy()
            else:
                blended_i = alpha * chunk_i + (1.0 - alpha) * raw_i[:m]
                blended_q = alpha * chunk_q + (1.0 - alpha) * raw_q[:m]

            self._out_i = self._out_i[m:]
            self._out_q = self._out_q[m:]
            self._trim_raw_archive(out_start + m)

            out_bytes = iq_to_bytes(blended_i, blended_q)
            self.samples_out += m
            return out_bytes

        out_bytes = iq_to_bytes(self._out_i, self._out_q)
        self.samples_out += len(self._out_i)
        self._out_i = np.empty(0, dtype=np.float32)
        self._out_q = np.empty(0, dtype=np.float32)
        return out_bytes

    def _drain_sample_queue(self) -> None:
        """Vectorised window extraction and batched inference.

        Uses ``np.lib.stride_tricks.sliding_window_view`` to extract **all**
        available windows from the sample queue in a single NumPy call
        (zero-copy views), then runs inference in mini-batches of ``B``.

        Replaces the former Python ``while`` loop + ``_run_batch()`` pair:
          • No per-window Python overhead — all window positions are computed
            in one NumPy stride operation.
          • Contiguous batch arrays are built with ``np.empty`` + slice
            assignment, avoiding the cost of repeated ``np.stack`` calls.
          • OLA outputs are collected into a pre-allocated batch array and
            concatenated once per mini-batch, halving allocator pressure.

        Queue trimming
        --------------
        After processing ``n_windows`` windows the first ``n_windows * H``
        samples of the queue have been fully consumed.  The tail
        ``[n_windows * H :]`` is kept for the next ``feed()`` call (it
        provides the overlap prefix for the upcoming window).
        ``n_windows * H ≤ q_len`` is guaranteed when ``H ≤ N`` (always true).
        """
        N, H, B = self.N, self.H, self.B
        q_len = len(self._q_i)
        if q_len < N:
            return

        n_windows = (q_len - N) // H + 1

        # Zero-copy stride views: shape (n_windows, N)
        view_i = np.lib.stride_tricks.sliding_window_view(self._q_i, N)[::H]
        view_q = np.lib.stride_tricks.sliding_window_view(self._q_q, N)[::H]

        for batch_start in range(0, n_windows, B):
            batch_end = min(batch_start + B, n_windows)
            b_size = batch_end - batch_start

            # Build contiguous (b_size, 2, N) tensor — one copy only
            batch_x = np.empty((b_size, 2, N), dtype=np.float32)
            batch_x[:, 0, :] = view_i[batch_start:batch_end]
            batch_x[:, 1, :] = view_q[batch_start:batch_end]

            # DC removal + RMS normalization — required when the model was
            # trained with SupervisedIQDataset(normalise=True).
            #
            # DC removal mirrors extract_labels.py which subtracts the
            # full-burst mean (~60 M samples) before saving label pairs.
            # Using the global warmup estimate avoids the distortion caused
            # by per-window subtraction: an ADS-B pulse inflates the 240-sample
            # window mean and causes over-removal that corrupts the signal shape.
            #
            # During warmup (first ~500 K samples) we fall back to per-window
            # subtraction as a conservative approximation; inference may miss
            # some frames but will not produce false structure.
            if self.normalize_windows:
                if self._dc_ready:
                    batch_x[:, 0, :] -= self._dc_i
                    batch_x[:, 1, :] -= self._dc_q
                else:
                    # Warmup fallback — per-window mean (less accurate)
                    batch_x[:, 0, :] -= batch_x[:, 0, :].mean(axis=1, keepdims=True)
                    batch_x[:, 1, :] -= batch_x[:, 1, :].mean(axis=1, keepdims=True)

                rms = (
                    np.sqrt(np.mean(batch_x ** 2, axis=(1, 2), keepdims=True))
                    + 1e-9
                )  # (b_size, 1, 1)
                batch_x = batch_x / rms

            with torch.no_grad():
                y_np = (
                    self.model(torch.from_numpy(batch_x).to(self.device))
                    .cpu()
                    .numpy()
                )  # (b_size, 2, N)

            if self.normalize_windows:
                y_np = y_np * rms  # restore original amplitude scale

            # Pre-allocate output arrays and fill from OLA push
            out_i_buf = np.empty((b_size, H), dtype=np.float32)
            out_q_buf = np.empty((b_size, H), dtype=np.float32)
            for b in range(b_size):
                out_i_buf[b], out_q_buf[b] = self._ola.push(
                    y_np[b, 0], y_np[b, 1]
                )

            self._enqueue_output(out_i_buf.ravel(), out_q_buf.ravel())
            self.n_inferences += 1

        # Trim consumed samples; keep the overlap prefix for the next feed()
        self._q_i = self._q_i[n_windows * H:].copy()
        self._q_q = self._q_q[n_windows * H:].copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, raw_bytes: bytes) -> bytes:
        """
        Process a chunk of raw RTL-SDR bytes.

        Handles:
          • Odd-length chunks (byte-alignment across reads).
          • Conversion uint8 → float32 → model → float32 → uint8.
          • OLA synthesis for smooth frame boundaries.

        Returns
        -------
        Processed uint8 bytes ready for stdout (may be empty if the batch
        accumulator has not yet been flushed).
        """
        # ── Byte alignment ───────────────────────────────────────────────
        # RTL-SDR pairs are 2 bytes (I, Q).  If the previous read left an
        # orphaned byte, prepend it before processing.
        combined = self._byte_buf + raw_bytes
        if len(combined) % 2 != 0:
            self._byte_buf = combined[-1:]
            combined = combined[:-1]
        else:
            self._byte_buf = b""

        if not combined:
            return b""

        # ── Normalise and extend sample queue ────────────────────────────
        new_i, new_q = bytes_to_iq(combined)
        self._q_i = np.concatenate([self._q_i, new_i])
        self._q_q = np.concatenate([self._q_q, new_q])
        self.samples_in += len(new_i)
        if self._blend_enabled:
            self._append_raw_archive(new_i, new_q)

        # ── DC warmup accumulation ────────────────────────────────────────
        # Accumulate raw samples until we have enough to form a reliable
        # hardware-bias estimate.  This mirrors extract_labels.py which uses
        # the whole-burst mean (~60 M samples).  We only need ~500 K because
        # the DC offset is stable within a capture session.
        if self.normalize_windows and not self._dc_ready:
            self._dc_warmup_buf_i.append(new_i)
            self._dc_warmup_buf_q.append(new_q)
            self._dc_warmup_n += len(new_i)
            if self._dc_warmup_n >= self._dc_warmup_target:
                all_i = np.concatenate(self._dc_warmup_buf_i)
                all_q = np.concatenate(self._dc_warmup_buf_q)
                self._dc_i = float(all_i.mean())
                self._dc_q = float(all_q.mean())
                self._dc_ready = True
                self._dc_warmup_buf_i.clear()
                self._dc_warmup_buf_q.clear()
                print(
                    f"[live_bridge] DC warmup complete: "
                    f"dc_i={self._dc_i:.6f}  dc_q={self._dc_q:.6f}  "
                    f"({self._dc_warmup_n:,} samples)",
                    file=sys.stderr,
                )

        # ── Drain queue into windows / batches ───────────────────────────
        self._drain_sample_queue()

        # ── Return any queued output (may take several blend passes) ───
        out = bytearray()
        while True:
            chunk = self._emit_pending_output()
            if not chunk:
                break
            out.extend(chunk)
        return bytes(out)

    def flush(self) -> bytes:
        """
        Drain all remaining windows and the OLA tail at end-of-stream.

        Call this once after the input is exhausted to recover the last
        (window_size − hop_size) output samples.
        """
        # Drain all remaining full windows (vectorised path handles batching)
        self._drain_sample_queue()

        # Force-flush the OLA remainder
        tail_i, tail_q = self._ola.flush_remaining()
        self._enqueue_output(tail_i, tail_q)

        out = bytearray()
        while True:
            chunk = self._emit_pending_output()
            if not chunk:
                break
            out.extend(chunk)

        # With blend, discard OLA tail samples beyond samples_in
        if self._blend_enabled:
            self._out_i = np.empty(0, dtype=np.float32)
            self._out_q = np.empty(0, dtype=np.float32)

        return bytes(out)

    def throughput_stats(self) -> dict:
        """Return current throughput statistics."""
        elapsed = time.perf_counter() - self._t_start
        return {
            "elapsed_s":     elapsed,
            "samples_in":    self.samples_in,
            "samples_out":   self.samples_out,
            "n_batches":     self.n_inferences,
            "mbps_in":       (self.samples_in * 2) / (elapsed * 1e6 + 1e-9),
            "realtime_ratio": (self.samples_in / (elapsed * 2e6 + 1e-9)),
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _get_device(device_str: Optional[str], threads: Optional[int] = None) -> torch.device:
    """Select the best available compute device with cross-platform fallback.

    Auto-detection priority: explicit arg → CUDA → MPS (Apple Silicon) → CPU.

    On CPU (including Raspberry Pi 4/5), the function sets PyTorch's intra-op
    thread count so all physical cores are utilised:
      • If ``threads`` is given, that value is used directly.
      • Otherwise the count defaults to ``os.cpu_count()`` (4 on RPi 4).

    Platform behaviour
    ------------------
    macOS + Apple Silicon : auto-selects MPS; CPU threads are not modified.
    Linux ARM64 (RPi 4/5) : MPS/CUDA unavailable → CPU, threads tuned to 4.
    Linux x86_64 + NVIDIA : auto-selects CUDA; CPU threads not modified.
    """
    if device_str:
        dev = torch.device(device_str)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if dev.type == "cpu":
        n_threads = threads if threads is not None else (os.cpu_count() or 1)
        torch.set_num_threads(n_threads)
        print(
            f"[live_bridge] CPU device — using {n_threads} intra-op thread(s)  "
            f"[{platform.machine()} / {platform.system()}]",
            file=sys.stderr,
        )

    return dev


def _setup_sigpipe() -> None:
    """Restore default SIGPIPE so a closed downstream pipe exits cleanly."""
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IQ denoising bridge: RTL-SDR raw bytes → IQAutoencoder → dump1090",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live piping (Linux):
  rtl_sdr -f 1090000000 -s 2000000 - | \\
      python live_bridge.py --ckpt checkpoints/best.pt | \\
      dump1090 --ifile /dev/stdin --raw

  # macOS (named FIFO):
  mkfifo /tmp/iq_pipe
  python live_bridge.py --ckpt checkpoints/best.pt < recorded.bin > /tmp/iq_pipe &
  dump1090 --ifile /tmp/iq_pipe --raw

  # Post-process a recorded file:
  python live_bridge.py --ckpt checkpoints/best.pt \\
      --input recorded.bin --output clean.bin

  # GPU-accelerated, higher throughput (no overlap):
  python live_bridge.py --ckpt checkpoints/best.pt --hop 240 --batch-size 64 --device mps

  # Raspberry Pi 4 — TorchScript + all 4 CPU cores:
  python live_bridge.py --ckpt checkpoints/best.pt --torchscript --threads 4 --hop 240 --batch-size 16
""",
    )
    parser.add_argument(
        "--ckpt", default="checkpoints/best.pt",
        help="Path to trained model checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Input file path.  Omit to read from stdin (pipe mode).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path.  Omit to write to stdout (pipe mode).",
    )
    parser.add_argument(
        "--hop", type=int, default=DEFAULT_HOP,
        help=f"OLA hop size in samples (default {DEFAULT_HOP} = 50%% overlap). "
             "Larger hop → faster, less smooth.  Must divide WINDOW_SIZE=240.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help=f"Windows per inference call (default {DEFAULT_BATCH}). "
             "Larger batch → better throughput; smaller → lower latency.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: cpu | cuda | mps (default: auto-detect).",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="CPU intra-op thread count (default: os.cpu_count()). "
             "Set to 4 on Raspberry Pi 4 for best throughput.",
    )
    parser.add_argument(
        "--torchscript", action="store_true",
        help="Trace the model with torch.jit.trace before inference.  "
             "Reduces Python dispatch overhead; recommended for RPi CPU.",
    )
    parser.add_argument(
        "--passthrough", action="store_true",
        help="Skip the bridge entirely — uint8→float→uint8 per chunk only. "
             "Tests IQ conversion rounding; does not exercise OLA or windowing.",
    )
    parser.add_argument(
        "--identity", action="store_true",
        help="Run the full bridge (OLA, DC warmup, RMS norm/denorm) but replace "
             "the model with output=input.  Use with compare_decodings.py to "
             "verify the pipeline preserves decodable frames before blaming the model.",
    )
    parser.add_argument(
        "--blend", type=float, default=1.0, metavar="ALPHA",
        help="After inference, mix output with raw input in float IQ space: "
             "out = ALPHA*model + (1-ALPHA)*raw.  1.0 = pure model (default). "
             "0.05 gave best decode results on adsb_capture.bin with v4. "
             "The model still processes 100%% of the stream; ALPHA weights the mix.",
    )
    parser.add_argument(
        "--normalize-windows", action="store_true", default=None, dest="normalize_windows",
        help="Divide each inference window by its RMS before the model and restore "
             "afterwards.  Required when the checkpoint was trained with "
             "SupervisedIQDataset(normalise=True) — i.e. best_supervised.pt.  "
             "Auto-detected from the checkpoint's training_mode field if not set.",
    )
    parser.add_argument(
        "--no-normalize-windows", action="store_false", dest="normalize_windows",
        help="Disable per-window RMS normalization (use for synthetic-only checkpoints).",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING). Stats always go to stderr.",
    )
    args = parser.parse_args()

    if args.passthrough and args.identity:
        log.error("--passthrough and --identity are mutually exclusive.")
        sys.exit(1)

    if not 0.0 <= args.blend <= 1.0:
        log.error("--blend must be between 0.0 and 1.0 (got %.4f).", args.blend)
        sys.exit(1)

    if args.passthrough and args.blend < 1.0 - 1e-9:
        log.error("--blend is ignored in --passthrough mode (output is already raw).")
        sys.exit(1)

    # ── Logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    # ── SIGPIPE: exit cleanly when downstream closes the pipe ────────────────
    _setup_sigpipe()

    # ── Validate hop ─────────────────────────────────────────────────────────
    if WINDOW_SIZE % args.hop != 0:
        log.error("--hop %d does not divide WINDOW_SIZE=%d evenly.", args.hop, WINDOW_SIZE)
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        log.error("Checkpoint not found: %s", ckpt)
        log.error("Train a model first:  python evaluate.py --train-epochs 50")
        sys.exit(1)

    device = _get_device(args.device, threads=args.threads)
    print(f"[live_bridge] Loading model from {ckpt} on {device} …", file=sys.stderr)
    model = load_model(str(ckpt), device=str(device))
    cfg   = model.config
    print(
        f"[live_bridge] Model: base_channels={cfg.base_channels}  "
        f"depth={cfg.depth}  params={model.count_parameters():,}",
        file=sys.stderr,
    )

    # ── Auto-detect per-window normalization from checkpoint metadata ─────────
    # Checkpoints saved by train_supervised.py carry training_mode="supervised_real".
    # Those models were trained with unit-RMS-normalised inputs and require the
    # same normalization at inference time.  Synthetic checkpoints (best.pt) do
    # not have this key and must NOT be normalized.
    if args.normalize_windows is None:
        raw_ckpt = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        training_mode = raw_ckpt.get("training_mode", "synthetic")
        normalize_windows = (training_mode == "supervised_real")
        del raw_ckpt
    else:
        normalize_windows = args.normalize_windows

    print(
        f"[live_bridge] normalize_windows={normalize_windows}"
        + (" (auto-detected from checkpoint)" if args.normalize_windows is None else " (manual override)"),
        file=sys.stderr,
    )

    # ── Optional TorchScript compilation ─────────────────────────────────────
    # torch.jit.trace eliminates Python dispatch overhead for each Conv1d op,
    # which can give 1.5–2× CPU speedup on a Raspberry Pi.  The traced module
    # has the identical __call__ interface, so the rest of the code is unchanged.
    if args.identity:
        def _identity_forward(x: torch.Tensor) -> torch.Tensor:
            return x

        model.forward = _identity_forward  # type: ignore[method-assign]
        print(
            "[live_bridge] IDENTITY mode — full bridge active, model returns input.",
            file=sys.stderr,
        )

    if args.torchscript and not args.passthrough and not args.identity:
        example = torch.zeros(1, 2, WINDOW_SIZE, device=device)
        model = torch.jit.trace(model, example)
        print("[live_bridge] TorchScript: model traced successfully.", file=sys.stderr)

    print(
        f"[live_bridge] Pipeline: window={WINDOW_SIZE}  hop={args.hop}  "
        f"batch={args.batch_size}  overlap={100*(1 - args.hop/WINDOW_SIZE):.0f}%"
        + ("  [TorchScript]" if args.torchscript and not args.passthrough and not args.identity else ""),
        file=sys.stderr,
    )

    if args.passthrough:
        print("[live_bridge] PASSTHROUGH mode — bridge disabled (uint8 round-trip only).", file=sys.stderr)

    if args.blend < 1.0 - 1e-9 and not args.passthrough:
        print(
            f"[live_bridge] BLEND mode — out = {args.blend:.3f}*model + "
            f"{1.0 - args.blend:.3f}*raw  (model still runs on full stream).",
            file=sys.stderr,
        )

    # ── I/O setup ────────────────────────────────────────────────────────────
    in_stream  = open(args.input,  "rb") if args.input  else sys.stdin.buffer
    out_stream = open(args.output, "wb") if args.output else sys.stdout.buffer

    # Flush stdout immediately so downstream doesn't starve
    if not args.output:
        # Make stdout unbuffered (pipe mode): wrap in RawIOBase passthrough
        out_stream = io.FileIO(sys.stdout.fileno(), mode="wb", closefd=False)

    # ── Pipeline instantiation ───────────────────────────────────────────────
    bridge = IQStreamBridge(
        model=model,
        device=device,
        hop_size=args.hop,
        batch_size=args.batch_size,
        normalize_windows=normalize_windows,
        blend_alpha=args.blend,
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    last_stats = time.perf_counter()
    bytes_read = 0

    print("[live_bridge] Streaming started.  Ctrl+C to stop.", file=sys.stderr)

    try:
        while True:
            raw = in_stream.read(READ_CHUNK)
            if not raw:
                break   # EOF

            bytes_read += len(raw)

            if args.passthrough:
                # Round-trip uint8 → float → uint8 without model
                i_arr, q_arr = bytes_to_iq(raw if len(raw) % 2 == 0 else raw[:-1])
                out_chunk    = iq_to_bytes(i_arr, q_arr)
            else:
                out_chunk = bridge.feed(raw)

            if out_chunk:
                out_stream.write(out_chunk)

            # ── Periodic throughput stats on stderr ──────────────────────────
            now = time.perf_counter()
            if now - last_stats >= STATS_EVERY_S:
                s = bridge.throughput_stats()
                print(
                    f"[live_bridge] "
                    f"{s['mbps_in']:.2f} MB/s in  |  "
                    f"{s['n_batches']} batches  |  "
                    f"real-time ratio: {s['realtime_ratio']:.2f}×  "
                    f"({'OK' if s['realtime_ratio'] >= 1.0 else 'BEHIND — increase --hop or --batch-size'})",
                    file=sys.stderr,
                )
                last_stats = now

    except KeyboardInterrupt:
        print("\n[live_bridge] Interrupted.", file=sys.stderr)
    except BrokenPipeError:
        pass   # downstream closed — normal pipe exit
    finally:
        # Flush remaining OLA tail
        if not args.passthrough:
            tail = bridge.flush()
            if tail:
                try:
                    out_stream.write(tail)
                except BrokenPipeError:
                    pass

        if args.input:
            in_stream.close()
        if args.output:
            out_stream.close()

        if not args.passthrough:
            s = bridge.throughput_stats()
            print(
                f"[live_bridge] Done.  "
                f"Read {bytes_read/1e6:.2f} MB  |  "
                f"{s['samples_in']:,} samples in  →  {s['samples_out']:,} samples out  |  "
                f"{s['n_batches']} inference batches  |  "
                f"avg {s['mbps_in']:.2f} MB/s",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
