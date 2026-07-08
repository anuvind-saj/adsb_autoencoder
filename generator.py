"""
generator.py — Synthetic ADS-B IQ Data Generator
==================================================
Synthesizes realistic 1090 MHz ADS-B Mode-S frames as raw complex baseband
(IQ) data.  Every frame is built from first principles so that the resulting
tensors are mathematically equivalent to what an RTL-SDR would capture when
tuned to 1090 MHz (with an optional DC-spike-clearing IF offset applied).

Signal specification
--------------------
Modulation   : OOK  (On-Off Keying) carrier, PPM (Pulse-Position Modulation)
               for the data payload.
Frame format : Mode-S long squitter (112 data bits + 8 µs preamble).
Sample rate  : 2.0 MHz  →  0.5 µs per sample.
Frame length : 120 µs  =  240 samples.

Preamble timing (4 pulses, each exactly 0.5 µs / 1 sample wide)
----------------------------------------------------------------
  Pulse 1  :  0.0 – 0.5 µs   →  sample index 0
  Pulse 2  :  1.0 – 1.5 µs   →  sample index 2
  Pulse 3  :  3.5 – 4.0 µs   →  sample index 7
  Pulse 4  :  4.5 – 5.0 µs   →  sample index 9
  Silence  :  5.0 – 8.0 µs   →  samples 10-15

PPM data encoding (1 µs / bit = 2 samples per bit)
---------------------------------------------------
  Bit '1'  →  pulse in first  sample of the pair (bit-half 0)
  Bit '0'  →  pulse in second sample of the pair (bit-half 1)

Phase continuity model
----------------------
  φ(n)  =  2π · f_offset · n · T_s  +  φ_0
  The phase accumulator runs continuously for all 240 samples — including
  the silence windows.  When the OOK carrier is off, the amplitude envelope
  is zero, but φ keeps rotating.  This is the key geometric property that
  the downstream CNN autoencoder is trained to learn.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Physical / timing constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: float = 2.0e6           # Hz
SAMPLE_PERIOD: float = 1.0 / SAMPLE_RATE   # 0.5 µs

PREAMBLE_SAMPLES: int = 16           # 8 µs × 2 MHz
SAMPLES_PER_BIT: int = 2             # 1 µs × 2 MHz
NUM_DATA_BITS: int = 112             # Mode-S long frame
DATA_SAMPLES: int = NUM_DATA_BITS * SAMPLES_PER_BIT   # 224
FRAME_SAMPLES: int = PREAMBLE_SAMPLES + DATA_SAMPLES  # 240

# Preamble pulse sample indices within the 16-sample preamble block.
# These are the canonical positions derived from RTCA DO-260B timing.
PREAMBLE_PULSE_INDICES: Tuple[int, ...] = (0, 2, 7, 9)


# ---------------------------------------------------------------------------
# Signal parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalParams:
    """
    All knobs that define a single ADS-B transmission as seen at the receiver.

    Transmitter properties (intrinsic to each aircraft)
    ────────────────────────────────────────────────────
    amplitude      : Linear signal amplitude (before noise).  1.0 = full scale.
    f_offset       : Carrier frequency offset from 0 Hz in Hz.  Each aircraft has
                     its own oscillator, so each has a unique f_offset.  A real
                     RTL-SDR tuned 250 kHz above 1090 MHz would see the centre of
                     the passband at +250 kHz; individual aircraft drift ±tens of kHz
                     around that centre.  This is the PRIMARY BSS discriminant in a
                     collision: two signals at different f_offset values spiral on the
                     complex plane at different angular rates and produce a measurable
                     beating envelope.
    initial_phase  : Starting carrier phase in radians (uniform [0, 2π)).  Each
                     aircraft's oscillator starts at an unknown phase at t=0.
    bits           : 112-element int8 array for the PPM payload.  None → random.

    Receiver / channel properties (added by the capture hardware, NOT by the aircraft)
    ───────────────────────────────────────────────────────────────────────────────────
    snr_db         : Signal-to-Noise Ratio used by add_impairments().
    dc_offset_i    : Constant I-channel bias from RTL-SDR hardware leakage.
                     This models the DC spike at 0 Hz visible in the baseband
                     spectrum.  It is a property of the RECEIVER, not the
                     transmitter.  In single-signal scenarios it is bundled here
                     for convenience; in collision scenarios, pass it as the
                     receiver_dc_i argument to synthesize_collision() instead.
    dc_offset_q    : Constant Q-channel bias (same origin as dc_offset_i).
    """

    amplitude: float = 1.0
    f_offset: float = 0.0
    initial_phase: float = 0.0
    snr_db: float = 20.0
    dc_offset_i: float = 0.0
    dc_offset_q: float = 0.0
    bits: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Core generator class
# ---------------------------------------------------------------------------

class ADSBSignalGenerator:
    """
    Generates synthetic ADS-B IQ frames as (2, FRAME_SAMPLES) float32 arrays.

    Channel layout matches what PyTorch 1D-CNN expects:
        output[0, :]  — I (In-Phase)
        output[1, :]  — Q (Quadrature)

    Parameters
    ----------
    sample_rate : Capture sample rate in Hz.  Defaults to 2.0 MHz.
    rng_seed    : Seed for NumPy's default_rng.  Pass an integer for
                  reproducible output; leave None for random behaviour.
    """

    def __init__(
        self,
        sample_rate: float = SAMPLE_RATE,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.sample_period = 1.0 / self.sample_rate
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_bits(self) -> np.ndarray:
        return self.rng.integers(0, 2, size=NUM_DATA_BITS, dtype=np.int8)

    def _validate_bits(self, bits: Optional[np.ndarray]) -> np.ndarray:
        if bits is None:
            return self._random_bits()
        bits = np.asarray(bits, dtype=np.int8)
        if bits.shape != (NUM_DATA_BITS,):
            raise ValueError(
                f"bits must have exactly {NUM_DATA_BITS} elements, "
                f"got shape {bits.shape}."
            )
        return bits

    def _build_envelope(self, bits: np.ndarray) -> np.ndarray:
        """
        Construct the binary OOK amplitude envelope for a full frame.

        The envelope is 1.0 at each sample where the carrier is on and
        0.0 everywhere else.  Silence gaps are explicit zeros — the CNN
        must learn that zero amplitude does not mean zero phase.

        Returns
        -------
        envelope : float32 array, shape (FRAME_SAMPLES,)
        """
        envelope = np.zeros(FRAME_SAMPLES, dtype=np.float32)

        # Preamble: four fixed pulses
        for idx in PREAMBLE_PULSE_INDICES:
            envelope[idx] = 1.0

        # PPM data payload
        for i, bit in enumerate(bits):
            base = PREAMBLE_SAMPLES + i * SAMPLES_PER_BIT
            if bit == 1:
                envelope[base] = 1.0           # pulse in first half
            else:
                envelope[base + 1] = 1.0       # pulse in second half

        return envelope

    def _phase_track(self, f_offset: float, initial_phase: float) -> np.ndarray:
        """
        Compute the analytically continuous phase accumulator for all FRAME_SAMPLES.

        φ(n)  =  2π · f_offset · n · T_s  +  φ_0        n ∈ {0, 1, …, FRAME_SAMPLES−1}

        PHASE CONTINUITY CONTRACT
        ─────────────────────────
        The phase is defined by a single closed-form expression that is evaluated
        at every sample index, INCLUDING samples that fall inside silence gaps
        (where the OOK amplitude envelope is zero).

        This means: when the carrier turns back on after a silence window, its
        phase continues from exactly where it would have been had the carrier
        never switched off — as if the oscillator kept running invisibly.  The
        receiver captures no energy during silence, but the transmitter's local
        oscillator never stops.

        This "hidden phase rotation" is the key geometric property that the CNN
        autoencoder must learn:
          • On-state samples   →  lie on the unit circle at phase φ(n).
          • Off-state samples  →  lie at the origin (zero amplitude).
          • The ARC between consecutive on-state pulses is not arbitrary noise;
            it is a deterministic arc gap of exactly Δφ = 2π·f_offset·gap·T_s.

        float64 precision prevents accumulation error for large f_offset values.

        Returns
        -------
        phase : float64 array, shape (FRAME_SAMPLES,)
        """
        n = np.arange(FRAME_SAMPLES, dtype=np.float64)
        return 2.0 * np.pi * f_offset * n * self.sample_period + initial_phase

    # ------------------------------------------------------------------
    # Public synthesis API
    # ------------------------------------------------------------------

    def synthesize_clean(self, params: SignalParams) -> np.ndarray:
        """
        Synthesize a single noise-free ADS-B IQ frame.

        The output is the geometrically perfect representation of the signal
        on the complex plane.  It is the training *target* for the autoencoder:
        the model learns to reconstruct this from a noisy or collided input.

        Parameters
        ----------
        params : SignalParams

        Returns
        -------
        iq : float32 array, shape (2, FRAME_SAMPLES)
             Row 0 = I channel, row 1 = Q channel.
        """
        bits = self._validate_bits(params.bits)
        phase = self._phase_track(params.f_offset, params.initial_phase)
        envelope = self._build_envelope(bits)

        # Complex phasor with OOK envelope applied
        carrier = (params.amplitude * np.exp(1j * phase)).astype(np.complex64)
        iq_complex = carrier * envelope.astype(np.complex64)

        return np.stack([iq_complex.real, iq_complex.imag], axis=0)  # (2, N)

    def add_impairments(
        self,
        clean_iq: np.ndarray,
        snr_db: float = 20.0,
        dc_offset_i: float = 0.0,
        dc_offset_q: float = 0.0,
        phase_noise_rad: float = 0.02,
        freq_drift_hz_per_sample: float = 0.0,
    ) -> np.ndarray:
        """
        Add real-world RF impairments to a clean IQ frame.

        Impairments applied in order:
          1. Frequency drift  — a slow, linear phase walk (oscillator wander).
          2. Phase noise      — sample-to-sample Gaussian phase jitter.
          3. AWGN             — thermal + quantization noise floor.
          4. DC offset        — constant bias on I and Q (hardware spike at 0 Hz).

        Parameters
        ----------
        clean_iq              : (2, N) float32 — the ideal signal.
        snr_db                : Desired signal-to-noise ratio in dB.
                                Lower values = more noise (e.g. 10 dB is weak).
        dc_offset_i           : Constant bias on the I channel.
        dc_offset_q           : Constant bias on the Q channel.
        phase_noise_rad       : Std-dev of per-sample Gaussian phase jitter (rad).
                                Typical RTL-SDR: 0.01 – 0.05 rad.
        freq_drift_hz_per_sample: Rate of linear frequency drift.
                                A value of 0.5 Hz/sample ≈ 1 kHz/ms drift.

        Returns
        -------
        noisy_iq : float32 array, shape (2, N)
        """
        noisy = clean_iq.copy().astype(np.float32)
        N = noisy.shape[1]

        # ── 1. Frequency drift (linear phase ramp) ────────────────────────
        if freq_drift_hz_per_sample != 0.0:
            n = np.arange(N, dtype=np.float64)
            # Quadratic phase ramp: φ_drift(n) = π · drift_rate · n²
            drift_phase = np.pi * freq_drift_hz_per_sample * n ** 2 * self.sample_period
            rotation = np.exp(1j * drift_phase).astype(np.complex64)
            phasor = (noisy[0] + 1j * noisy[1]) * rotation
            noisy[0] = phasor.real
            noisy[1] = phasor.imag

        # ── 2. Phase noise (stochastic jitter) ───────────────────────────
        if phase_noise_rad > 0.0:
            jitter = self.rng.standard_normal(N).astype(np.float32) * phase_noise_rad
            phasor = (noisy[0] + 1j * noisy[1]) * np.exp(1j * jitter)
            noisy[0] = phasor.real
            noisy[1] = phasor.imag

        # ── 3. AWGN ───────────────────────────────────────────────────────
        signal_power = float(np.mean(noisy ** 2))
        if signal_power > 0.0:
            # Noise power split equally between I and Q
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise_std = float(np.sqrt(noise_power / 2.0))
            noisy[0] += self.rng.standard_normal(N).astype(np.float32) * noise_std
            noisy[1] += self.rng.standard_normal(N).astype(np.float32) * noise_std
        else:
            warnings.warn(
                "Signal power is zero — AWGN calibrated to unit power instead.",
                stacklevel=2,
            )
            noise_std = float(np.sqrt(0.5 / (10.0 ** (snr_db / 10.0))))
            noisy[0] += self.rng.standard_normal(N).astype(np.float32) * noise_std
            noisy[1] += self.rng.standard_normal(N).astype(np.float32) * noise_std

        # ── 4. DC offset ─────────────────────────────────────────────────
        noisy[0] += dc_offset_i
        noisy[1] += dc_offset_q

        return noisy

    def synthesize_noisy(self, params: SignalParams, **impairment_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper: synthesize a clean frame then add impairments.

        Returns
        -------
        (noisy_iq, clean_iq) : each float32 array, shape (2, FRAME_SAMPLES)
        """
        clean = self.synthesize_clean(params)
        noisy = self.add_impairments(
            clean,
            snr_db=params.snr_db,
            dc_offset_i=params.dc_offset_i,
            dc_offset_q=params.dc_offset_q,
            **impairment_kwargs,
        )
        return noisy, clean

    def synthesize_collision(
        self,
        params_a: SignalParams,
        params_b: SignalParams,
        time_offset_samples: int = 0,
        add_noise: bool = True,
        receiver_dc_i: float = 0.0,
        receiver_dc_q: float = 0.0,
        min_freq_separation_hz: float = 5_000.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a co-channel collision between two aircraft.

        PHYSICS OF THE INTERFERENCE / BEATING EFFECT
        ─────────────────────────────────────────────
        Two aircraft transmit on 1090 MHz with independent oscillators.  After
        mixing to baseband, each signal occupies a unique frequency offset:

            s_A(t) = A_a · rect(t) · exp(j·(2π·f_a·t + φ_a))
            s_B(t) = A_b · rect(t) · exp(j·(2π·f_b·t + φ_b))

        where rect(t) is the OOK envelope (1 during pulses, 0 during silence).
        The receiver captures their LINEAR SUPERPOSITION:

            r(t) = s_A(t) + s_B(t)

        Because f_a ≠ f_b, the two phasors rotate at different angular rates.
        During windows where both envelopes are 1, they alternately add
        constructively (|r| ≤ A_a + A_b) and destructively (|r| ≥ |A_a − A_b|)
        at the BEAT FREQUENCY:

            f_beat = |f_a − f_b|         (Hz)
            T_beat = 1 / f_beat          (seconds)

        This produces the "shaky combined vector" and oscillating magnitude
        envelope — the primary BSS discriminant the autoencoder exploits.  The
        network learns that the on-state samples of each source trace a SINGLE
        clean arc on the unit circle (constant |IQ| = amplitude, monotone phase),
        while the composite traces an erratic Lissajous figure.

        DC OFFSET — RECEIVER-LEVEL PARAMETER
        ──────────────────────────────────────
        The DC bias at 0 Hz is caused by RTL-SDR hardware self-mixing leakage.
        It belongs to the RECEIVER, not to any individual transmitter.  Pass it
        via receiver_dc_i / receiver_dc_q.  Do NOT use params_a.dc_offset_i or
        params_b.dc_offset_i here — those fields apply to single-signal calls
        (synthesize_noisy) where the distinction does not matter.

        Parameters
        ----------
        params_a             : Primary aircraft (training reconstruction target).
        params_b             : Interfering aircraft.
        time_offset_samples  : Integer sample delay of signal B relative to A.
                               0  = perfectly synchronised (hardest case).
                               >0 = B starts after A.
                               <0 = B's preamble overlaps A's data.
        add_noise            : Add AWGN to the composite.  A 3 dB SNR penalty is
                               applied because both signals contribute power.
        receiver_dc_i        : Fixed I-channel bias from RTL-SDR hardware leakage.
                               Applied to the COMPOSITE after superposition.
        receiver_dc_q        : Fixed Q-channel bias from RTL-SDR hardware leakage.
        min_freq_separation_hz: Minimum required |f_a − f_b|.  If the two params
                               have an insufficient frequency separation, the beating
                               effect will be slow or invisible within a single frame.
                               A warning is raised so the caller can fix their params.

        Returns
        -------
        (collision_iq, clean_a, clean_b) : each float32, shape (2, FRAME_SAMPLES)
            collision_iq : noisy composite received by the RTL-SDR.
            clean_a      : noise-free signal A (model reconstruction target).
            clean_b      : noise-free signal B, shifted by time_offset_samples.
        """
        # ── Frequency separation guard ────────────────────────────────────────
        freq_sep = abs(params_a.f_offset - params_b.f_offset)
        if freq_sep < min_freq_separation_hz:
            beat_period_samples = (
                (self.sample_rate / freq_sep) if freq_sep > 0 else float("inf")
            )
            warnings.warn(
                f"Collision frequency separation {freq_sep:.1f} Hz is below the "
                f"minimum {min_freq_separation_hz:.1f} Hz.  "
                f"Beat period ≈ {beat_period_samples:.1f} samples "
                f"({beat_period_samples * self.sample_period * 1e6:.1f} µs) — "
                f"the beating envelope may span many frames and be invisible to "
                f"the autoencoder within a single {FRAME_SAMPLES}-sample window.  "
                f"Set distinct f_offset values separated by at least "
                f"{min_freq_separation_hz:.0f} Hz for a visible beat cycle.",
                stacklevel=2,
            )

        clean_a = self.synthesize_clean(params_a)
        clean_b = self.synthesize_clean(params_b)

        # ── Shift signal B in time ────────────────────────────────────────────
        if time_offset_samples != 0:
            shifted_b = np.zeros_like(clean_b)
            off = int(time_offset_samples)
            if off > 0:
                shifted_b[:, off:] = clean_b[:, : FRAME_SAMPLES - off]
            else:  # off < 0
                shifted_b[:, : FRAME_SAMPLES + off] = clean_b[:, -off:]
        else:
            shifted_b = clean_b.copy()

        # ── Linear field superposition ────────────────────────────────────────
        # This is field addition (voltages add), NOT power addition.
        # The beating effect emerges automatically from this single line because
        # the two phasors rotate at different angular rates and their vector sum
        # oscillates in magnitude at f_beat = |f_a − f_b|.
        composite = clean_a + shifted_b

        # ── Receiver impairments ──────────────────────────────────────────────
        # DC offset is applied to the COMPOSITE — it is a single receiver artifact,
        # not a per-transmitter property.  AWGN carries a 3 dB penalty because
        # the combined signal power is ≈ P_a + P_b.
        if add_noise:
            collision_snr = min(params_a.snr_db, params_b.snr_db) - 3.0
            collision_iq = self.add_impairments(
                composite,
                snr_db=collision_snr,
                dc_offset_i=receiver_dc_i,
                dc_offset_q=receiver_dc_q,
            )
        else:
            # Even without AWGN, the DC bias (if any) must still be applied.
            collision_iq = composite.astype(np.float32)
            collision_iq[0] += receiver_dc_i
            collision_iq[1] += receiver_dc_q

        return collision_iq, clean_a, shifted_b

    # ------------------------------------------------------------------
    # Batch generation for PyTorch DataLoader
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_size: int,
        snr_db_range: Tuple[float, float] = (8.0, 30.0),
        f_offset_range: Tuple[float, float] = (-50_000.0, 50_000.0),
        dc_offset_mag_range: Tuple[float, float] = (0.0, 0.05),
        phase_noise_rad: float = 0.025,
        freq_drift_hz_per_sample: float = 0.0,
        include_collisions: bool = False,
        collision_fraction: float = 0.20,
        collision_time_offset_range: Tuple[int, int] = (-24, 24),
        collision_min_freq_sep_hz: float = 8_000.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of (noisy_input, clean_target) pairs.

        Designed to be called inside a PyTorch Dataset.__getitem__ or
        used to pre-generate an in-memory dataset for fast training.

        Parameters
        ----------
        batch_size                  : Number of samples.
        snr_db_range                : Uniform range for per-sample SNR (dB).
        f_offset_range              : Uniform range for carrier frequency offset (Hz).
                                      Mimics both direct-to-baseband and IF-offset capture.
        dc_offset_mag_range         : Absolute magnitude range for the RECEIVER DC bias.
                                      Sign is randomised independently per sample.
                                      This models RTL-SDR hardware leakage — applied once
                                      to the final composite, never per-transmitter.
        phase_noise_rad             : Fixed phase jitter std-dev applied to all samples.
        freq_drift_hz_per_sample    : Frequency drift rate (0 = off).
        include_collisions          : Mix collision scenarios into the batch.
        collision_fraction          : Fraction of the batch that are collisions [0, 1].
        collision_time_offset_range : Integer (min, max) sample delay for signal B.
                                      Spans ±12 µs at 2 MHz.
        collision_min_freq_sep_hz   : Minimum |f_a − f_b| enforced for collision pairs.
                                      Guarantees that at least one full beat cycle
                                      (constructive + destructive interference) is
                                      visible inside the 120 µs frame.
                                      Default 8 kHz → T_beat = 125 µs ≈ 1 frame.

        Returns
        -------
        noisy_batch : float32 array, shape (batch_size, 2, FRAME_SAMPLES)
        clean_batch : float32 array, shape (batch_size, 2, FRAME_SAMPLES)
                      For collision samples, clean_batch[i] is the primary aircraft's
                      clean signal — the desired BSS reconstruction target.
        """
        noisy_batch = np.empty((batch_size, 2, FRAME_SAMPLES), dtype=np.float32)
        clean_batch = np.empty((batch_size, 2, FRAME_SAMPLES), dtype=np.float32)

        f_lo, f_hi = float(f_offset_range[0]), float(f_offset_range[1])

        for i in range(batch_size):
            snr = float(self.rng.uniform(*snr_db_range))
            f_off = float(self.rng.uniform(f_lo, f_hi))
            dc_mag_i = float(self.rng.uniform(*dc_offset_mag_range))
            dc_mag_q = float(self.rng.uniform(*dc_offset_mag_range))
            dc_i = dc_mag_i * float(self.rng.choice([-1.0, 1.0]))
            dc_q = dc_mag_q * float(self.rng.choice([-1.0, 1.0]))
            phi0 = float(self.rng.uniform(0.0, 2.0 * np.pi))

            if include_collisions and float(self.rng.random()) < collision_fraction:
                # ── Collision scenario ────────────────────────────────────────
                # Draw params for aircraft B independently.  Enforce minimum
                # frequency separation so a visible beating envelope is guaranteed
                # within the frame duration.
                amp_b = float(self.rng.uniform(0.3, 1.0))
                phi0_b = float(self.rng.uniform(0.0, 2.0 * np.pi))

                # Reject-resample f_off_b until it is far enough from f_off_a.
                # The half-range from which B is drawn is clamped to ensure the
                # separation is achievable within f_offset_range.
                for _ in range(64):
                    f_off_b = float(self.rng.uniform(f_lo, f_hi))
                    if abs(f_off - f_off_b) >= collision_min_freq_sep_hz:
                        break
                else:
                    # Fallback: place B exactly collision_min_freq_sep_hz above A.
                    f_off_b = f_off + collision_min_freq_sep_hz
                    if f_off_b > f_hi:
                        f_off_b = f_off - collision_min_freq_sep_hz

                t_min, t_max = collision_time_offset_range
                t_off = int(self.rng.integers(t_min, t_max + 1))

                pa = SignalParams(amplitude=1.0,  f_offset=f_off,   initial_phase=phi0,   snr_db=snr)
                pb = SignalParams(amplitude=amp_b, f_offset=f_off_b, initial_phase=phi0_b, snr_db=snr)
                collision_noisy, ca, _ = self.synthesize_collision(
                    pa, pb,
                    time_offset_samples=t_off,
                    receiver_dc_i=dc_i,      # DC is a receiver property, applied once
                    receiver_dc_q=dc_q,      # to the composite, not per-signal
                    min_freq_separation_hz=collision_min_freq_sep_hz,
                )
                noisy_batch[i] = collision_noisy
                clean_batch[i] = ca
            else:
                # ── Single-signal denoising scenario ─────────────────────────
                params = SignalParams(
                    amplitude=1.0,
                    f_offset=f_off,
                    initial_phase=phi0,
                    snr_db=snr,
                    dc_offset_i=dc_i,
                    dc_offset_q=dc_q,
                )
                clean_iq = self.synthesize_clean(params)
                noisy_iq = self.add_impairments(
                    clean_iq,
                    snr_db=snr,
                    dc_offset_i=dc_i,
                    dc_offset_q=dc_q,
                    phase_noise_rad=phase_noise_rad,
                    freq_drift_hz_per_sample=freq_drift_hz_per_sample,
                )
                noisy_batch[i] = noisy_iq
                clean_batch[i] = clean_iq

        return noisy_batch, clean_batch


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_iq_frame(
    clean_iq: np.ndarray,
    noisy_iq: Optional[np.ndarray] = None,
    collision_iq: Optional[np.ndarray] = None,
    title: str = "ADS-B IQ Frame",
    save_path: Optional[str] = None,
    sample_rate: float = SAMPLE_RATE,
) -> None:
    """
    Visualise the I, Q channels and the IQ constellation of a generated frame.

    Layout (2 × 2 grid)
    -------------------
    [0,0] I channel waveform      [0,1] Q channel waveform
    [1,0] IQ constellation        [1,1] Magnitude envelope

    Preamble / data boundary is marked on the magnitude panel.
    A unit-circle reference is drawn on the constellation panel to emphasise
    the geometric arc that valid on-state samples should lie on.

    Parameters
    ----------
    clean_iq     : (2, N) float32 — the ground-truth noise-free signal.
    noisy_iq     : (2, N) float32 — noisy version to overlay (optional).
    collision_iq : (2, N) float32 — collision composite to overlay (optional).
    title        : Figure super-title.
    save_path    : File path to save the figure; None = do not save.
    sample_rate  : Used to compute the time axis in µs.
    """
    sp = 1.0 / sample_rate
    N = clean_iq.shape[1]
    t_us = np.arange(N) * sp * 1e6   # time axis in microseconds

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    C_CLEAN = "#1f77b4"
    C_NOISY = "#ff7f0e"
    C_COLL  = "#d62728"
    ALPHA_OV = 0.65

    # ── [0,0] I channel ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_us, clean_iq[0], color=C_CLEAN, lw=1.8, label="Clean I", zorder=3)
    if noisy_iq is not None:
        ax.plot(t_us, noisy_iq[0], color=C_NOISY, lw=0.9, alpha=ALPHA_OV, label="Noisy I")
    if collision_iq is not None:
        ax.plot(t_us, collision_iq[0], color=C_COLL, lw=0.9, alpha=ALPHA_OV, label="Collision I")
    ax.axvline(x=PREAMBLE_SAMPLES * sp * 1e6, color="purple", ls=":", lw=1.0)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title("I Channel (In-Phase)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── [0,1] Q channel ───────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_us, clean_iq[1], color="#2ca02c", lw=1.8, label="Clean Q", zorder=3)
    if noisy_iq is not None:
        ax.plot(t_us, noisy_iq[1], color=C_NOISY, lw=0.9, alpha=ALPHA_OV, label="Noisy Q")
    if collision_iq is not None:
        ax.plot(t_us, collision_iq[1], color=C_COLL, lw=0.9, alpha=ALPHA_OV, label="Collision Q")
    ax.axvline(x=PREAMBLE_SAMPLES * sp * 1e6, color="purple", ls=":", lw=1.0)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Q Channel (Quadrature)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── [1,0] IQ constellation ────────────────────────────────────────────
    ax = axes[1, 0]
    # Unit-circle reference (on-state samples should lie near this arc)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.25, label="Unit circle")

    if noisy_iq is not None:
        ax.scatter(noisy_iq[0], noisy_iq[1], s=5, color=C_NOISY, alpha=0.45, label="Noisy")
    if collision_iq is not None:
        ax.scatter(collision_iq[0], collision_iq[1], s=5, color=C_COLL, alpha=0.45, label="Collision")

    # Draw the clean arc with a sequential colour map to show phase direction
    on_mask = clean_iq[0] ** 2 + clean_iq[1] ** 2 > 0.01   # on-state samples
    if on_mask.sum() > 1:
        sc = ax.scatter(
            clean_iq[0, on_mask],
            clean_iq[1, on_mask],
            c=np.where(on_mask)[0],
            cmap="cool",
            s=18,
            zorder=5,
            label="Clean (on-state)",
        )
        plt.colorbar(sc, ax=ax, label="Sample index")

    ax.scatter(clean_iq[0, ~on_mask], clean_iq[1, ~on_mask],
               s=4, color="lightgray", alpha=0.4, label="Clean (off-state)", zorder=2)
    ax.set_xlabel("I (In-Phase)")
    ax.set_ylabel("Q (Quadrature)")
    ax.set_title("IQ Constellation\n(colour = time, arc = phase rotation)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── [1,1] Magnitude envelope ──────────────────────────────────────────
    ax = axes[1, 1]
    mag_clean = np.sqrt(clean_iq[0] ** 2 + clean_iq[1] ** 2)
    ax.plot(t_us, mag_clean, color=C_CLEAN, lw=1.8, label="Clean |IQ|", zorder=3)
    if noisy_iq is not None:
        mag_noisy = np.sqrt(noisy_iq[0] ** 2 + noisy_iq[1] ** 2)
        ax.plot(t_us, mag_noisy, color=C_NOISY, lw=0.9, alpha=ALPHA_OV, label="Noisy |IQ|")
    if collision_iq is not None:
        mag_coll = np.sqrt(collision_iq[0] ** 2 + collision_iq[1] ** 2)
        ax.plot(t_us, mag_coll, color=C_COLL, lw=0.9, alpha=ALPHA_OV, label="Collision |IQ|")

    boundary_us = PREAMBLE_SAMPLES * sp * 1e6
    ax.axvline(x=boundary_us, color="purple", ls=":", lw=1.1,
               label=f"Preamble | Data ({boundary_us:.1f} µs)")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("|I + jQ|")
    ax.set_title("Magnitude Envelope √(I² + Q²)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")

    plt.show()


def summarise_frame(label: str, iq: np.ndarray) -> None:
    """Print a compact diagnostic summary of a (2, N) IQ array."""
    mag = np.sqrt(iq[0] ** 2 + iq[1] ** 2)
    print(
        f"  {label:<22s}  shape={iq.shape}  "
        f"I∈[{iq[0].min():+.3f}, {iq[0].max():+.3f}]  "
        f"Q∈[{iq[1].min():+.3f}, {iq[1].max():+.3f}]  "
        f"|IQ|_max={mag.max():.3f}  |IQ|_mean={mag.mean():.4f}"
    )


def compute_beat_info(
    params_a: SignalParams,
    params_b: SignalParams,
    sample_rate: float = SAMPLE_RATE,
) -> dict:
    """
    Analytically predict the interference / beating properties of a collision.

    When two OOK-modulated carriers at different frequency offsets are received
    simultaneously, their superimposed phasors produce an oscillating composite
    magnitude.  For samples where BOTH signals are on simultaneously:

        |r(t)| = |A_a·exp(j·φ_a(t)) + A_b·exp(j·φ_b(t))|

    As (φ_a − φ_b) sweeps through 2π, |r| oscillates between:
        maximum (constructive): A_a + A_b
        minimum (destructive):  |A_a − A_b|

    The sweep rate is the BEAT FREQUENCY:
        f_beat = |f_a − f_b|   (Hz)
        T_beat = 1 / f_beat    (seconds)

    For the beating to be visible within a single 120 µs ADS-B frame, at least
    a fraction of one beat cycle should fit inside the frame.  Recommended:
        f_beat > 1 / (FRAME_SAMPLES × T_s) ≈ 8.3 kHz

    Parameters
    ----------
    params_a, params_b : SignalParams for the two colliding aircraft.
    sample_rate        : Receiver sample rate in Hz.

    Returns
    -------
    dict with keys
        f_a_hz              : Frequency offset of aircraft A (Hz).
        f_b_hz              : Frequency offset of aircraft B (Hz).
        freq_separation_hz  : |f_a − f_b| (Hz) — equals beat frequency.
        beat_period_us      : Beating period in microseconds.
        beat_period_samples : Beating period in samples (float).
        beats_per_frame     : How many beat cycles fit in one 120 µs frame.
        envelope_max        : Peak composite magnitude = A_a + A_b.
        envelope_min        : Trough magnitude = |A_a − A_b|.
        modulation_depth    : (max − min) / max  ∈ [0, 1].
                              1.0 = perfect cancellation (A_a == A_b).
                              0.0 = no beating (one signal dominates).
    """
    sep = abs(params_a.f_offset - params_b.f_offset)
    t_s = 1.0 / sample_rate
    frame_duration_s = FRAME_SAMPLES * t_s
    beat_period_s = (1.0 / sep) if sep > 0 else float("inf")
    env_max = params_a.amplitude + params_b.amplitude
    env_min = abs(params_a.amplitude - params_b.amplitude)
    return {
        "f_a_hz":              params_a.f_offset,
        "f_b_hz":              params_b.f_offset,
        "freq_separation_hz":  sep,
        "beat_period_us":      beat_period_s * 1e6,
        "beat_period_samples": beat_period_s * sample_rate,
        "beats_per_frame":     frame_duration_s / beat_period_s if sep > 0 else 0.0,
        "envelope_max":        env_max,
        "envelope_min":        env_min,
        "modulation_depth":    (env_max - env_min) / env_max if env_max > 0 else 0.0,
    }


def plot_collision_analysis(
    params_a: SignalParams,
    params_b: SignalParams,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    collision_iq: np.ndarray,
    title: str = "Collision Analysis",
    save_path: Optional[str] = None,
    sample_rate: float = SAMPLE_RATE,
) -> None:
    """
    Four-panel deep-dive visualisation of the three physical phenomena.

    Panel layout
    ────────────
    [0,0] Magnitude envelopes of A, B, and composite — beating clearly visible.
          Analytical beat envelope (constructive peak / destructive trough) is
          overlaid in dashed lines.
    [0,1] IQ constellations — each source traces a single clean arc at its own
          radius; the composite traces an erratic Lissajous figure.
    [1,0] Unwrapped phase vs time — demonstrates PHASE CONTINUITY.
          φ_A and φ_B are straight lines (slope = 2π·f_offset), proving the
          oscillators never stop between pulses.  The composite phase is plotted
          only for on-state samples and jumps erratically, showing why BSS is hard.
    [1,1] DC offset demonstration — I-channel close-up of the composite with and
          without DC bias, showing the constant vertical shift the autoencoder
          must learn to remove.

    Parameters
    ----------
    params_a, params_b : SignalParams used to generate the signals.
    clean_a, clean_b   : (2, N) float32 — individual clean signals.
    collision_iq       : (2, N) float32 — the noisy received composite.
    title              : Figure title.
    save_path          : Optional PNG save path.
    sample_rate        : Receiver sample rate (Hz).
    """
    sp = 1.0 / sample_rate
    N = clean_a.shape[1]
    t_us = np.arange(N) * sp * 1e6

    beat = compute_beat_info(params_a, params_b, sample_rate)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    C_A    = "#1f77b4"
    C_B    = "#2ca02c"
    C_COLL = "#d62728"
    C_BEAT = "#9467bd"

    # ── [0,0] Magnitude envelopes + analytical beat bounds ────────────────
    ax = axes[0, 0]
    mag_a    = np.sqrt(clean_a[0]**2    + clean_a[1]**2)
    mag_b    = np.sqrt(clean_b[0]**2    + clean_b[1]**2)
    mag_coll = np.sqrt(collision_iq[0]**2 + collision_iq[1]**2)

    ax.plot(t_us, mag_a,    color=C_A,    lw=1.4, label=f"Aircraft A  |  f={params_a.f_offset/1e3:+.1f} kHz, A={params_a.amplitude:.2f}")
    ax.plot(t_us, mag_b,    color=C_B,    lw=1.4, label=f"Aircraft B  |  f={params_b.f_offset/1e3:+.1f} kHz, A={params_b.amplitude:.2f}")
    ax.plot(t_us, mag_coll, color=C_COLL, lw=1.0, alpha=0.8, label="Composite (received)")

    # Analytical beat envelope — only meaningful where both signals overlap
    both_on = (mag_a > 0.1) & (mag_b > 0.1)
    if both_on.sum() > 1:
        ax.axhline(y=beat["envelope_max"], color=C_BEAT, ls="--", lw=1.0, alpha=0.7,
                   label=f"Beat max = {beat['envelope_max']:.2f}  (constructive)")
        ax.axhline(y=beat["envelope_min"], color=C_BEAT, ls=":",  lw=1.0, alpha=0.7,
                   label=f"Beat min = {beat['envelope_min']:.2f}  (destructive)")

    ax.axvline(x=PREAMBLE_SAMPLES * sp * 1e6, color="purple", ls=":", lw=0.8)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("|IQ|")
    ax.set_title(
        f"Magnitude Envelopes — Beating Effect\n"
        f"f_beat = {beat['freq_separation_hz']/1e3:.1f} kHz  |  "
        f"T_beat = {beat['beat_period_us']:.1f} µs  |  "
        f"{beat['beats_per_frame']:.2f} cycles/frame  |  "
        f"depth = {beat['modulation_depth']*100:.0f} %"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── [0,1] IQ constellations — A arc, B arc, composite Lissajous ───────
    ax = axes[0, 1]
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.6, alpha=0.2)

    # Show on-state samples for A, B coloured by time index, and composite
    mask_a = mag_a > 0.1
    mask_b = mag_b > 0.1
    if mask_a.sum() > 1:
        ax.scatter(clean_a[0, mask_a], clean_a[1, mask_a],
                   c=np.where(mask_a)[0], cmap="Blues", s=14, zorder=4,
                   label="Aircraft A (on-state)", vmin=0, vmax=N)
    if mask_b.sum() > 1:
        ax.scatter(clean_b[0, mask_b], clean_b[1, mask_b],
                   c=np.where(mask_b)[0], cmap="Greens", s=14, zorder=4,
                   label="Aircraft B (on-state)", vmin=0, vmax=N)
    ax.scatter(collision_iq[0], collision_iq[1],
               s=4, color=C_COLL, alpha=0.35, label="Composite (Lissajous)", zorder=2)

    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("IQ Constellation — Two Phase Arcs vs Composite Lissajous")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── [1,0] Unwrapped phase — PHASE CONTINUITY ─────────────────────────
    ax = axes[1, 0]
    n_arr = np.arange(N, dtype=np.float64)

    # Analytical (ground-truth) phase lines — straight, never reset
    phi_a_analytical = (2 * np.pi * params_a.f_offset * n_arr * sp + params_a.initial_phase)
    phi_b_analytical = (2 * np.pi * params_b.f_offset * n_arr * sp + params_b.initial_phase)

    ax.plot(t_us, phi_a_analytical / (2 * np.pi), color=C_A, lw=1.6,
            label=f"φ_A analytical  (slope = {params_a.f_offset/1e3:+.1f} kHz)")
    ax.plot(t_us, phi_b_analytical / (2 * np.pi), color=C_B, lw=1.6,
            label=f"φ_B analytical  (slope = {params_b.f_offset/1e3:+.1f} kHz)")

    # Measured phase of composite — only at on-state samples, unwrapped
    phasor_coll = collision_iq[0] + 1j * collision_iq[1]
    mag_composite = np.abs(phasor_coll)
    on_composite = mag_composite > (0.1 * mag_composite.max() if mag_composite.max() > 0 else 0.1)
    if on_composite.sum() > 1:
        phi_coll_on = np.unwrap(np.angle(phasor_coll[on_composite]))
        ax.scatter(t_us[on_composite], phi_coll_on / (2 * np.pi),
                   s=5, color=C_COLL, alpha=0.5, label="φ_composite (on-state, unwrapped)")

    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Phase (cycles = radians / 2π)")
    ax.set_title(
        "Unwrapped Phase vs Time — Phase Continuity\n"
        "Straight lines = constant f_offset; phase never resets during silence"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=PREAMBLE_SAMPLES * sp * 1e6, color="purple", ls=":", lw=0.8)

    # ── [1,1] DC offset close-up — I channel first 40 samples ────────────
    ax = axes[1, 1]
    composite_clean = (clean_a + clean_b).astype(np.float32)
    t_zoom = t_us[:40]
    ax.plot(t_zoom, composite_clean[0, :40], color=C_A, lw=1.8,
            label="Composite I  (no DC bias)")
    ax.plot(t_zoom, collision_iq[0, :40], color=C_COLL, lw=1.2, alpha=0.85,
            label=f"Received I  (DC_i={collision_iq[0].mean() - composite_clean[0].mean():+.4f})")
    dc_level = float(collision_iq[0].mean() - composite_clean[0].mean())
    ax.axhline(y=dc_level, color="gray", ls="--", lw=0.9, alpha=0.7,
               label=f"Estimated DC bias ≈ {dc_level:+.4f}")
    ax.axhline(y=0.0, color="black", ls=":", lw=0.5, alpha=0.4)
    ax.set_xlabel("Time (µs)  — first 20 µs (40 samples)")
    ax.set_ylabel("I Amplitude")
    ax.set_title("DC Offset — RTL-SDR Hardware Leakage\nConstant bias shifts the IQ origin away from (0, 0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Three demonstration scenarios:
      1. Denoising — single aircraft with AWGN, DC offset, and phase noise.
      2. Collision  — two aircraft with distinct frequency offsets and amplitudes.
      3. Batch      — 512-sample batch including 20 % collision scenarios.
    """
    gen = ADSBSignalGenerator(rng_seed=42)

    sep = "=" * 62

    # ── Scenario 1: Denoising ─────────────────────────────────────────────
    print(sep)
    print("Scenario 1 — Single aircraft (denoising task)")
    print(sep)
    p1 = SignalParams(
        amplitude=1.0,
        f_offset=25_000.0,          # 25 kHz offset (e.g. Low-IF capture)
        initial_phase=np.pi / 4,
        snr_db=15.0,
        dc_offset_i=+0.05,
        dc_offset_q=-0.03,
    )
    clean1 = gen.synthesize_clean(p1)
    noisy1 = gen.add_impairments(
        clean1,
        snr_db=p1.snr_db,
        dc_offset_i=p1.dc_offset_i,
        dc_offset_q=p1.dc_offset_q,
        phase_noise_rad=0.03,
    )
    print(f"  Frame duration  : {FRAME_SAMPLES} samples  "
          f"({FRAME_SAMPLES * SAMPLE_PERIOD * 1e6:.1f} µs)")
    summarise_frame("Clean IQ", clean1)
    summarise_frame("Noisy IQ (SNR=15 dB)", noisy1)
    plot_iq_frame(
        clean1,
        noisy_iq=noisy1,
        title="Scenario 1 — Single Aircraft  |  SNR = 15 dB, f_off = +25 kHz",
        save_path="adsb_scenario1_single.png",
    )

    # ── Scenario 2: Co-channel collision ──────────────────────────────────
    print(f"\n{sep}")
    print("Scenario 2 — Co-channel collision  (beating + phase continuity + DC offset)")
    print(sep)
    #
    # Two aircraft with:
    #   • Independent amplitudes  (A=1.0, B=0.65)  →  modulation depth < 1.0
    #   • Independent initial phases  (φ_a=0°, φ_b=60°)  →  unknown phase offset
    #   • Independent frequency offsets  (+30 kHz, −12 kHz)  →  f_beat = 42 kHz
    #     T_beat = 23.8 µs ≈ 47.6 samples → ~2.5 beat cycles per 120 µs frame
    #   • Receiver DC bias (not per-aircraft!)
    #
    pa = SignalParams(amplitude=1.0,  f_offset=+30_000.0, initial_phase=0.0,       snr_db=22.0)
    pb = SignalParams(amplitude=0.65, f_offset=-12_000.0, initial_phase=np.pi / 3, snr_db=22.0)

    beat_info = compute_beat_info(pa, pb)
    print(f"  Beat frequency     : {beat_info['freq_separation_hz']/1e3:.1f} kHz")
    print(f"  Beat period        : {beat_info['beat_period_us']:.1f} µs  "
          f"({beat_info['beat_period_samples']:.1f} samples)")
    print(f"  Beats per frame    : {beat_info['beats_per_frame']:.2f}")
    print(f"  Envelope max/min   : {beat_info['envelope_max']:.2f} / {beat_info['envelope_min']:.2f}")
    print(f"  Modulation depth   : {beat_info['modulation_depth']*100:.0f} %")

    collision, ca, cb = gen.synthesize_collision(
        pa, pb,
        time_offset_samples=3,
        receiver_dc_i=+0.06,   # RTL-SDR hardware DC spike — applied ONCE to composite
        receiver_dc_q=-0.04,
        min_freq_separation_hz=8_000.0,
    )
    summarise_frame("Aircraft A (clean)", ca)
    summarise_frame("Aircraft B (clean)", cb)
    summarise_frame("Collision composite", collision)

    plot_collision_analysis(
        pa, pb, ca, cb, collision,
        title=(
            "Scenario 2 — Co-channel Collision Analysis\n"
            "A: +30 kHz @ 0°, amp=1.0  |  B: −12 kHz @ 60°, amp=0.65  |  "
            "DC_i=+0.06, DC_q=−0.04  |  offset=3 samples"
        ),
        save_path="adsb_scenario2_collision_analysis.png",
    )

    # ── Scenario 3: Batch generation ─────────────────────────────────────
    print(f"\n{sep}")
    print("Scenario 3 — Batch generation (512 samples, 20 % collisions)")
    print(sep)
    noisy_b, clean_b = gen.generate_batch(
        batch_size=512,
        include_collisions=True,
        collision_fraction=0.20,
    )
    print(f"  noisy_batch shape : {noisy_b.shape}  dtype={noisy_b.dtype}")
    print(f"  clean_batch shape : {clean_b.shape}  dtype={clean_b.dtype}")
    print(f"  noisy mean power  : {(noisy_b ** 2).mean():.5f}")
    print(f"  clean mean power  : {(clean_b ** 2).mean():.5f}")


if __name__ == "__main__":
    _demo()
