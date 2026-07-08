# WHITEPAPER BLUEPRINT
## Phase-Aware 1D-CNN U-Net Autoencoder for Blind Source Separation and Denoising of Co-Channel ADS-B Mode S IQ Signals

> **Document Status:** Engineering Blueprint — Technical facts, code-verified metrics, and design decisions captured for future formal paper compilation.
> **All values herein are code-verified against the actual implementation files unless explicitly noted as estimates.**

---

## TABLE OF CONTENTS

1. [Executive Summary & End Goals](#1-executive-summary--end-goals)
2. [Physical System Constraints & Hardware Anomalies](#2-physical-system-constraints--hardware-anomalies)
3. [Data Generation Methodology](#3-data-generation-methodology)
4. [Network Architecture Specification](#4-network-architecture-specification)
5. [Complex Phase-Aware Loss Function](#5-complex-phase-aware-loss-function)
6. [Results & Milestones Achieved](#6-results--milestones-achieved)
7. [Known Limitations & Edge Cognizance](#7-known-limitations--edge-cognizance)
8. [Next Phases & Evolutionary Roadmap](#8-next-phases--evolutionary-roadmap)
9. [System File Inventory](#9-system-file-inventory)

---

## 1. EXECUTIVE SUMMARY & END GOALS

### 1.1 Primary Objective

This project develops a **Phase-Aware 1D-CNN U-Net Autoencoder** (`IQAutoencoder`) that operates directly on raw complex baseband IQ samples to achieve two primary goals simultaneously:

1. **Denoising** — Remove thermal noise, RTL-SDR hardware artefacts (DC offset, phase jitter, frequency drift), and quantisation distortion from received ADS-B signals while preserving the sub-microsecond pulse timing required for Mode-S decoding.

2. **Blind Source Separation (BSS) of Co-Channel Collisions** — Reconstruct the primary aircraft's Mode-S transmission from the superposition of two overlapping signals, exploiting the distinct carrier frequency offsets (and resulting beating/phase-sliding envelope) as the principal discriminant — without any prior knowledge of the interfering aircraft's identity or signal parameters.

The resulting denoised IQ stream is written back to standard RTL-SDR binary format (`uint8` interleaved I, Q) and piped directly into legacy decoders (e.g., `dump1090-fa`) on edge hardware (Raspberry Pi 4), maximising valid CRC-passing frame throughput without modifying the downstream decoder.

### 1.2 Why This Problem Is Hard

Standard ADS-B decoders (`dump1090`, `dump978`) operate on magnitude-only envelopes after squaring and low-pass filtering the IQ stream. This destroys all phase information. For a single clean signal, this is adequate. For co-channel collisions — which occur with probability proportional to traffic density squared — the magnitude-only view is irrecoverable: destructive phase interference appears as a silent gap, constructive interference appears as amplitude overshoot, and both corrupt the PPM bit positions used for message reconstruction.

By retaining and processing both channels \(I\) and \(Q\) through the full pipeline, this system has access to geometric information invisible to magnitude-only decoders: the **spiral trajectory** of each aircraft's phasor on the complex plane, its **angular velocity** (proportional to its unique oscillator frequency offset), and the **Lissajous beating envelope** that encodes the presence of exactly two superimposed sources.

### 1.3 Target Audience

- Signal processing engineers working with SDR pipelines, ADS-B/ATC protocols, and embedded RF systems.
- Deep learning researchers in time-series reconstruction, blind source separation, and physics-informed neural network design.
- Aviation safety and surveillance engineers evaluating AI-augmented reception for high-density airspace.

### 1.4 Hardware Context

| Component | Specification |
|---|---|
| Antenna | Custom 8-segment Coaxial Collinear (CoCo), tuned 1090 MHz |
| Receiver | RTL-SDR (RTL2832U chipset) |
| Capture Pi | Raspberry Pi 3B (collector), Raspberry Pi 4 (inference) |
| Development machine | Apple Silicon MacBook (MPS-accelerated training) |
| Storage | MinIO S3-compatible object store on Raspberry Pi 4 + 1 TB SSD |
| Antennas deployed | `rpi_east`, `rpi_west` — two independent orientations |

---

## 2. PHYSICAL SYSTEM CONSTRAINTS & HARDWARE ANOMALIES

### 2.1 Why Standard Decoders Fail at Collisions

ADS-B Mode-S uses **Pulse Position Modulation (PPM)** on an OOK carrier. A decoder must:
1. Detect the 8 µs preamble (4 pulses at 0, 1, 3.5, 4.5 µs).
2. For each of the 112 data bits, compare the energy in the first vs. second half of each 1 µs bit window to determine '1' or '0'.
3. Validate the 24-bit CRC appended to the payload.

When two aircraft transmit simultaneously, the received magnitude envelope is:

$$|r(t)| = \left| A_a \cdot e^{j\phi_a(t)} + A_b \cdot e^{j\phi_b(t)} \right| \neq |A_a \cdot e^{j\phi_a(t)}| + |A_b \cdot e^{j\phi_b(t)}|}$$

The magnitude of the sum is **not** the sum of the magnitudes. Depending on the instantaneous phase difference \(\Delta\phi(t) = \phi_a(t) - \phi_b(t)\):
- **Constructive** (\(\Delta\phi \approx 0\)): \(|r| \approx A_a + A_b\) — pulse appears double-strength.
- **Destructive** (\(\Delta\phi \approx \pi\)): \(|r| \approx |A_a - A_b|\) — pulse may vanish entirely.

A standard threshold-based preamble detector receives garbled pulse positions and the downstream CRC check fails, dropping the message entirely. In busy airspace, collision rates of 5–15% are reported in literature; near major airports this can exceed 30%.

### 2.2 RTL-SDR Hardware Anomalies Modelled

#### 2.2.1 DC Offset

RTL-SDR dongles use a **direct-conversion (zero-IF) architecture**. The local oscillator leaks into the RF front end and self-mixes, creating a constant bias at 0 Hz in the baseband spectrum visible as a sharp spike in the FFT. In the time domain this manifests as:

$$I_{received}(n) = I_{signal}(n) + \delta_I$$
$$Q_{received}(n) = Q_{signal}(n) + \delta_Q$$

where \(\delta_I, \delta_Q \in [0.01, 0.05]\) (normalised units) are hardware-specific constants. In `generator.py`, this is modelled as a **receiver-level property** (not per-transmitter) applied to the composite signal after superposition (`receiver_dc_i`, `receiver_dc_q` parameters in `synthesize_collision()`).

In the real-world training pipeline (`extract_labels.py`, `real_dataset.py`), DC removal is applied before any processing:
```python
I -= float(I.mean())
Q -= float(Q.mean())
```

#### 2.2.2 Thermal Noise (AWGN)

RTL-SDR noise is modelled as **Additive White Gaussian Noise** split equally between I and Q channels:

$$\sigma_{noise} = \sqrt{\frac{P_{signal}}{2 \cdot 10^{SNR_{dB}/10}}}$$

Training SNR range: \(SNR \in [8.0, 25.0]\) dB for synthetic data; real-world gain rotation at \(\{28, 32, 36, 40, 44\}\) dB provides additional SNR diversity in captured data.

#### 2.2.3 Phase Jitter

Per-sample Gaussian phase noise with standard deviation `phase_noise_rad = 0.025` rad is applied to simulate RTL-SDR oscillator instability:

$$s_{jitter}(n) = s(n) \cdot e^{j \cdot \mathcal{N}(0,\ \sigma^2_{jitter})}$$

#### 2.2.4 Frequency Drift

Linear phase ramp modelling oscillator wander:

$$\phi_{drift}(n) = \pi \cdot \dot{f}_{drift} \cdot n^2 \cdot T_s$$

where \(\dot{f}_{drift}\) is in Hz/sample. This causes a slow Doppler-like shift during the 120 µs frame duration.

### 2.3 The Sampling Rate Selection: 2.0 MHz vs. 2.4 MHz

**This is a critical design constraint with mathematical justification.**

ADS-B Mode-S PPM data encoding uses a **1 µs bit cell** divided into two half-cells of **0.5 µs** each:

| Rate | Sample period | Samples per PPM bit | Samples per preamble |
|---|---|---|---|
| 2.0 MHz | 0.5 µs | **2 (exact)** | **16 (exact)** |
| 2.4 MHz | 0.417 µs | 2.4 (fractional) | 19.2 (fractional) |

At **2.0 MHz**, every single sample occupies exactly one PPM bit-half. This means:
- The preamble occupies exactly **16 samples** (indices 0–15).
- Each data bit occupies exactly **2 samples** (samples 0 and 1 of each pair).
- The full Mode-S long frame occupies exactly **240 samples**: \(16 + 112 \times 2 = 240\).
- The resulting tensor shape `(B, 2, 240)` achieves **perfect geometric alignment**: every pulse-on sample maps to exactly one index in the tensor with no fractional sub-sample ambiguity.

At 2.4 MHz, bit boundaries fall between samples, requiring interpolation. The resulting non-integer sample indices break the 1:1 geometric mapping between tensor positions and PPM bit decisions — the model would need to learn fractional alignment implicitly, significantly increasing the representational burden.

**The 2.0 MHz rate is not a compromise — it is the unique rate at which the ADS-B PPM grid and the neural network tensor grid are *identical*.**

```python
# From generator.py — these constants define the mathematical contract:
SAMPLE_RATE: float = 2.0e6           # Hz
SAMPLE_PERIOD: float = 1.0 / SAMPLE_RATE   # 0.5 µs = exactly 1 PPM bit-half
PREAMBLE_SAMPLES: int = 16           # 8 µs × 2 MHz
SAMPLES_PER_BIT: int = 2             # 1 µs × 2 MHz
NUM_DATA_BITS: int = 112             # Mode-S long squitter
DATA_SAMPLES: int = 224              # 112 × 2
FRAME_SAMPLES: int = 240             # 16 + 224
```

---

## 3. DATA GENERATION METHODOLOGY

### 3.1 Overview

`generator.py` (`ADSBSignalGenerator`) synthesises fully controllable ADS-B IQ frames from first principles. This serves as the mathematical ground truth for Phase 1 (synthetic pre-training) and provides the clean reconstruction targets in Phase 2 (supervised real-world fine-tuning via re-synthesis).

### 3.2 Frame Construction Pipeline

#### Step 1: PPM Envelope

```
Preamble pulses (from RTCA DO-260B):
  Sample index 0  → pulse (0.0 – 0.5 µs)
  Sample index 2  → pulse (1.0 – 1.5 µs)
  Sample index 7  → pulse (3.5 – 4.0 µs)
  Sample index 9  → pulse (4.5 – 5.0 µs)
  Samples 10–15   → silence

PPM data (per bit i, base = 16 + 2i):
  Bit = 1 → envelope[base]   = 1.0  (pulse in first  half)
  Bit = 0 → envelope[base+1] = 1.0  (pulse in second half)
```

#### Step 2: Phase Accumulator (Phase Continuity Contract)

$$\phi(n) = 2\pi \cdot f_{offset} \cdot n \cdot T_s + \phi_0, \quad n \in \{0, 1, \ldots, 239\}$$

**Critical property:** The phase accumulator runs continuously over ALL 240 samples, including silence windows where the OOK envelope is zero. When the carrier turns off during a silence gap, the oscillator keeps rotating invisibly. When it turns back on, it resumes from the exact phase it would have reached — not from an arbitrary reset.

This means:
- On-state samples lie on a **continuous arc** on the complex plane at radius `amplitude`.
- Off-state samples lie at the **origin** (magnitude = 0).
- The **arc gap** between two consecutive on-state pulses carries information: it encodes \(2\pi \cdot f_{offset} \cdot \Delta n \cdot T_s\), which is the primary feature the encoder uses to infer the carrier's frequency offset.

#### Step 3: IQ Synthesis

$$\text{IQ}(n) = A \cdot e^{j\phi(n)} \cdot \text{envelope}(n)$$

Stored as two real channels:
```python
iq = np.stack([iq_complex.real, iq_complex.imag], axis=0)  # (2, 240)
```

#### Step 4: Impairment Application (Order-Dependent)

Applied in the following order to match RTL-SDR signal chain physics:
1. **Frequency drift** (linear phase ramp — oscillator wander)
2. **Phase jitter** (per-sample Gaussian perturbation)
3. **AWGN** (thermal + quantisation noise, calibrated to `snr_db`)
4. **DC offset** (hardware self-mixing leakage, constant bias)

### 3.3 Collision Synthesis: The Physics of Beating

Two aircraft transmit on 1090 MHz simultaneously with independent oscillators. After baseband mixing, each occupies a unique frequency offset:

$$s_A(t) = A_a \cdot \text{rect}(t) \cdot e^{j(2\pi f_a t + \phi_a)}$$
$$s_B(t) = A_b \cdot \text{rect}(t) \cdot e^{j(2\pi f_b t + \phi_b)}$$

The receiver captures the **linear superposition** (field addition — voltages, not powers):

$$r(t) = s_A(t) + s_B(t)$$

In discrete implementation:
```python
# From generator.py synthesize_collision():
composite = clean_a + shifted_b   # linear field superposition
```

#### The Beating Effect (BSS Discriminant)

Because \(f_a \neq f_b\), the two phasors rotate at different angular velocities. During windows where both envelopes are active, their vector sum oscillates at the **beat frequency**:

$$f_{beat} = |f_a - f_b| \quad \text{(Hz)}$$

$$T_{beat} = \frac{1}{f_{beat}} \quad \text{(seconds)}$$

The beat manifests as an amplitude modulation envelope on top of the combined signal:

$$|r(t)|_{on} \in \left[|A_a - A_b|,\ A_a + A_b\right]$$

oscillating at \(f_{beat}\). In the training data, source frequency offsets are drawn from \(f_{offset} \in [-50\,000, +50\,000]\) Hz, with a minimum separation of \(8\,000\) Hz enforced by `min_freq_separation_hz` to ensure at least one visible beat cycle within the 240-sample (120 µs) frame window:

$$\text{Beat samples} = \frac{f_s}{f_{beat}} = \frac{2 \times 10^6}{f_{beat}}$$

At 8,000 Hz separation: \(250\) samples/beat — one full cycle per frame. At 50,000 Hz: \(40\) samples/beat — six cycles per frame, rapid beating clearly visible.

#### Training Pair Structure for Collision Scenarios

```
Input  (noisy):  collision_iq  — composite + AWGN + DC offset  (B, 2, 240)
Target (clean):  clean_a       — noise-free signal A only        (B, 2, 240)
```

The model is trained to project the composite Lissajous trajectory back onto the clean spiral of signal A, performing implicit single-source selection.

#### Collision Fraction in Training

`collision_fraction = 0.25` — 25% of each training batch consists of two-signal collision scenarios. The remaining 75% are single-signal denoising examples (SNR ∈ [8, 25] dB).

### 3.4 Hybrid Collision Augmentation (Semi-Real Mixtures)

#### 3.4.1 Motivation

The `extract_labels.py` pipeline can only produce supervised pairs for frames that pass the 24-bit Mode-S CRC. A real co-channel collision almost always corrupts the CRC and is silently discarded. This creates a structural gap: the supervised dataset contains zero real collision examples, meaning the model's BSS capability is exclusively determined by the synthetic pre-training (Section 3.3) and does not benefit from the 406 GB real-world capture corpus.

The hybrid augmentation technique bridges this gap without requiring any additional hardware or data collection.

#### 3.4.2 Technique

Implemented as `collision_augment` in `SupervisedIQDataset` (`supervised_dataset.py`). On each `__getitem__` call, with probability `p` (default `p = 0.30`), a synthetic second-aircraft signal is linearly superimposed onto the real noisy input:

$$x_{augmented}(n) = x_{real,A}(n) + s_{synthetic,B}(n)$$

The clean target remains signal A only:

$$y_{target}(n) = \hat{s}_{clean,A}(n) \quad \text{(re-synthesized, unchanged)}$$

The model is therefore trained to project the composite back onto signal A — identical to the real-world collision task, but with an authentic noise floor from the real hardware capture.

#### 3.4.3 Interferer Signal B: Calibration Parameters

| Parameter | Distribution | Rationale |
|---|---|---|
| Amplitude B | \(U(\alpha_{min}, \alpha_{max}) \times \hat{A}_A\) | Calibrated relative to signal A's estimated amplitude \(\hat{A}_A\) from `amp_est` field in `.npz`; covers weak (0.3×, −10 dB) through dominant (1.5×, +3.5 dB) interferers |
| Frequency offset B | Drawn from \(f_{offset\_range}\) with gap \(\geq f_{min\_sep}\) around \(\hat{f}_A\) | `collision_min_freq_sep_hz = 8,000` Hz ensures at least one full beating cycle within the 240-sample window; excludes the band \([\hat{f}_A - 8k, \hat{f}_A + 8k]\) Hz |
| Initial phase B | \(U(0, 2\pi)\) | Unknown oscillator phase |
| Payload bits B | Random 112 bits | Collision partner identity is irrelevant — only signal geometry matters for BSS |
| Time offset | \(U_{int}(-24, +24)\) samples | Covers ±12 µs: fully synchronised collisions, partial preamble overlaps, and data-only overlaps |

#### 3.4.4 Design Considerations During Calibration

Three non-obvious constraints had to be resolved before the augmentation produced physically plausible training examples:

**Consideration 1 — Amplitude must be relative, not absolute.**
Signal B cannot be drawn from a fixed amplitude range (e.g., `[0.3, 1.5]`) because the real captures have varying absolute power levels depending on aircraft distance and the SDR gain setting in use at capture time (the gain rotation between 28–44 dB creates a ~16 dB span in received power levels across files). Setting B's amplitude to a fixed value would produce unrealistic scenarios where a "strong" interferer is weaker than the real signal's noise floor at low-gain captures, or where a "weak" interferer dominates at high-gain captures.
The solution is to anchor B's amplitude to signal A's **estimated amplitude** (`amp_est` from the `.npz` file, derived from the preamble pulse height at label extraction time):
$$A_B = U(\alpha_{min},\ \alpha_{max}) \times \hat{A}_A$$
This makes the SNR ratio between the two sources independent of the absolute gain level, preserving physical realism across the full gain diversity of the training corpus.

**Consideration 2 — Frequency offset B must be constrained away from A's offset, not just randomised.**
A naïve draw of `f_offset_B` from the full training range `[−50, +50]` kHz would occasionally place B very close to A's estimated frequency offset `f_offset_est`. When `|f_B − f_A| ≲ 1` kHz, the beat period exceeds 2,000 samples — far longer than the 240-sample frame window. The model would see a collision with no measurable beating and would have no geometric discriminant to separate the two sources. The minimum 8 kHz separation forces beat periods ≤ 250 samples (≤ 1 frame), ensuring the angular velocity difference is visible within the training window:
$$f_{beat} = |f_B - f_A| \geq 8\,000 \text{ Hz} \implies T_{beat} \leq 250 \text{ samples} = 1 \text{ frame}$$
The allowed draw range is split into two sub-intervals — below and above the exclusion band — and sampled proportionally by bandwidth so neither side is systematically over-represented.

**Consideration 3 — Time offset must cover partial preamble overlaps, not just fully synchronised cases.**
If time offset is fixed at zero, the model trains exclusively on the case where both aircraft start transmitting simultaneously — which is the hardest case for decoders but not the most common in practice. Real collisions typically occur when one aircraft's data payload overlaps another's preamble or data at an arbitrary alignment. The integer time offset drawn uniformly from `[−24, +24]` samples covers: (a) signal B's preamble overlapping signal A's preamble (|offset| ≤ 16), (b) signal B's preamble landing inside signal A's data payload (offset ∈ [16, 24]), and (c) signal B starting slightly before A (negative offsets — B's data overlaps A's preamble). This breadth prevents the model from learning a collision-specific feature that only applies to the synchronised case.

#### 3.4.5 Known Approximation: The Clean Interferer Caveat

Signal B is generated using `generator.py`'s `synthesize_clean()` — a mathematically perfect ADS-B signal with no AWGN, no oscillator jitter, no frequency drift, and perfectly rectangular pulse envelopes. In a real co-channel collision, both aircraft arrive with their own independent noise contributions:

$$r(t) = (s_A(t) + n_A(t)) + (s_B(t) + n_B(t)) = s_A(t) + s_B(t) + n_{total}(t)$$

In the augmentation, signal B arrives noise-free, so the total noise in the training input is only \(n_A(t)\) — the real hardware noise of capture A:

$$x_{augmented}(t) = (s_A(t) + n_A(t)) + s_B(t)$$

The consequence is that the model may subtly learn to treat any signal with zero noise floor as "the interferer to remove" rather than purely relying on frequency-offset discriminant. In practice this is a minor second-order effect for two reasons:
1. The **primary BSS discriminant** — the angular velocity difference on the complex plane — is correctly and fully represented by `synthesize_clean()`. The beating envelope, the phase arc curvature, and the temporal structure of the collision all depend on the frequency offset, not the noise level.
2. The **real noise on signal A** — which the model must preserve through the denoising step — is already well-covered by the single-aircraft pairs that make up the other 70% of each batch. The model learns what authentic RTL-SDR noise looks like from those examples and is not misled by B's clean representation.

The clean-interferer approximation would only become a meaningful issue if training were conducted **exclusively** on collision-augmented pairs with no clean single-aircraft examples. With `collision_augment = 0.30`, clean pairs dominate training (70% of batch) and provide the noise distribution reference.

This approximation is explicitly noted as a known limitation rather than a defect — it would be resolved if a dual-synchronised SDR setup were available to capture real collisions with ground-truth knowledge of both signals.

#### 3.4.6 Why This Preserves Physical Correctness

Signal B is synthesised using `generator.py`'s `synthesize_clean()` — the same mathematical model used to produce all synthetic pre-training data. The superposition is a linear field addition (voltage, not power), exactly as it occurs at the antenna:

```python
# From supervised_dataset.py _add_synthetic_interferer():
return (noisy_a + clean_b).astype(np.float32)
```

The noise floor in the augmented input is the real RTL-SDR hardware noise from the original capture of signal A. The primary BSS discriminant (different angular velocity on the complex plane) is fully and correctly represented.

#### 3.4.7 Data Multiplication

With `collision_augment = 0.30`, approximately 30% of each training batch consists of hybrid collision examples. Since the augmentation is stochastic and unique per epoch (the RNG seed combines a global state with the sample index and epoch-dependent noise), each of the ~400K real pairs can produce many distinct collision variants across training epochs, effectively extending the BSS training corpus by an order of magnitude with zero additional data collection.

#### 3.4.8 Training Commands

**v2 (initial — 30% clean interferer, retroactively identified as flawed):**
```bash
python train_supervised.py \
    --labels-dir "/Volumes/KBMA SSD/labels" \
    --warm-start  checkpoints/best_supervised.pt \
    --ckpt-out    checkpoints/best_supervised_v2.pt \
    --epochs 50 \
    --collision-augment 0.30
```
*Result: val loss 2.208 — best training loss achieved, but produced real-world regression (8 valid frames vs. v1's 116). Root cause: clean synthetic interferer + 30% rate caused over-suppression. See §7.7.*

**v3 (corrected — 10% noisy interferer, recommended configuration):**
```bash
python train_supervised.py \
    --labels-dir "/Volumes/KBMA SSD/labels" \
    --warm-start  checkpoints/best_supervised.pt \
    --ckpt-out    checkpoints/best_supervised_v3.pt \
    --epochs 50 \
    --collision-augment 0.10 \
    --interferer-snr-db 20.0
```
*Fixes: (1) rate reduced to 10% so denoising objective dominates; (2) 20 dB AWGN added to interferer prevents model exploiting "clean = synthetic = suppress" shortcut.*

The `collision_augment` and `interferer_snr_db` are both saved to the checkpoint metadata for traceability:
```python
{"training_mode": "supervised_real", "collision_augment": 0.10, "interferer_snr_db": 20.0, ...}
```

---

## 4. NETWORK ARCHITECTURE SPECIFICATION

### 4.1 Architecture Family

**U-Net 1D Convolutional Autoencoder** (`IQAutoencoder` in `model.py`).

Design rationale: U-Net was chosen over a vanilla encoder-decoder because:
1. **Skip connections** preserve the exact sample-level timing of preamble pulses. Without them, the decoder must hallucinate pulse positions from the compressed latent alone — causing blurring at the ±0.5 µs scale that destroys PPM bit decisions.
2. **Multi-scale feature extraction** — the encoder hierarchy captures both local pulse shape (high-resolution stages) and global frame-level phase trajectory / beating envelope (low-resolution stages).
3. **1D convolutions** are the natural operator for this problem: the signal is a 1D time series in two channels (I, Q), and convolutional weight sharing makes the network **translation-equivariant** — it detects the same pulse pattern regardless of where in the frame it appears.

### 4.2 Configuration Parameters (`AutoencoderConfig`)

| Parameter | Value | Description |
|---|---|---|
| `seq_len` | 240 | Sequence length in samples |
| `in_channels` | 2 | I and Q |
| `base_channels` | 64 | Channels at encoder stage 1 (production model) |
| `depth` | 4 | Number of encode/decode stages |
| `bottleneck_layers` | 2 | Conv layers in bottleneck |
| `leaky_slope` | 0.2 | LeakyReLU negative slope |
| `dropout` | 0.0 | Disabled in production |

**Constraint:** `seq_len` must be divisible by \(2^{depth} = 16\). At 240 = 16 × 15, this is satisfied exactly.

### 4.3 Encoder

Four `_EncoderBlock` stages, each containing:
1. `Conv1d(in_ch, out_ch, kernel=3, stride=2, padding=1)` → halves time axis
2. `BatchNorm1d(out_ch)`
3. `LeakyReLU(0.2)`
4. `Conv1d(out_ch, out_ch, kernel=3, stride=1, padding=1)` → deepen at new resolution
5. `BatchNorm1d(out_ch)`
6. `LeakyReLU(0.2)`

**Dimensional flow (base_channels=64):**

| Stage | Input shape | Output shape | Channel schedule |
|---|---|---|---|
| Input | `(B, 2, 240)` | — | 2 channels |
| Encoder 0 | `(B, 2, 240)` | `(B, 64, 120)` | 2 → 64 |
| Encoder 1 | `(B, 64, 120)` | `(B, 128, 60)` | 64 → 128 |
| Encoder 2 | `(B, 128, 60)` | `(B, 256, 30)` | 128 → 256 |
| Encoder 3 | `(B, 256, 30)` | `(B, 512, 15)` | 256 → 512 |

**Channel schedule:** `[2, 64, 128, 256, 512]`

Each encoder output is stored as a **skip connection** before the next downsampling.

### 4.4 Bottleneck

Two `Conv1d(512, 512, kernel=3, padding=1)` layers operating at `(B, 512, 15)`:
- Fixed spatial resolution — no further compression.
- 15 time steps at this resolution each represent **16 samples** in the original frame (120 µs / 15 = 8 µs per latent step — exactly one preamble pulse width).
- Allows the network to integrate evidence from the entire frame context to resolve which of the two beating sources should be reconstructed.

> **The Bottleneck as Noise Purgatory:** At `(B, 512, 15)`, the model operates at a 16× temporal compression. Gaussian noise, which has no temporal structure, cannot survive this compression — its energy is spread across the 512 feature channels and averages to near-zero in each latent dimension. Structured signal features (the beating envelope, the phase arc gradient, the preamble timing fingerprint) do survive because they are low-entropy and spatially coherent. The bottleneck is where physical signal from thermal noise separation occurs.

### 4.5 Decoder

Three `_DecoderBlock` stages (one skip per stage from encoder reverse order), each containing:
1. `ConvTranspose1d(in_ch, out_ch, kernel=3, stride=2, padding=1, output_padding=1)` → doubles time axis
2. `BatchNorm1d(out_ch)`
3. `LeakyReLU(0.2)`
4. **Skip concatenation:** `cat([upsampled, skip], dim=1)` — channel axis
5. `Conv1d(out_ch + skip_ch, out_ch, kernel=3, padding=1)` → fuse
6. `BatchNorm1d(out_ch)`
7. `LeakyReLU(0.2)`

**Decoder dimensional flow:**

| Stage | Input shape | Skip shape | Output shape |
|---|---|---|---|
| Decoder 0 | `(B, 512, 15)` | `(B, 256, 30)` | `(B, 256, 30)` |
| Decoder 1 | `(B, 256, 30)` | `(B, 128, 60)` | `(B, 128, 60)` |
| Decoder 2 | `(B, 128, 60)` | `(B, 64, 120)` | `(B, 64, 120)` |
| Final upsample | `(B, 64, 120)` | *(no skip)* | `(B, 64, 240)` |
| Output head | `(B, 64, 240)` | — | `(B, 2, 240)` |

**Why the final upsample has no skip:** The input tensor has only 2 channels (I, Q), which carries insufficient feature depth to form a meaningful skip at this scale. A skip here would inject raw noisy input directly into the output, partially defeating the denoising.

### 4.6 Output Head

```python
nn.Conv1d(base_channels, 2, kernel_size=1)  # 1×1 pointwise convolution
```

No final activation. The network outputs **unbounded real-valued IQ coordinates** — the model is not constrained to output values on the unit circle. The loss function provides the geometric constraint implicitly.

### 4.7 Weight Initialisation

- Conv / ConvTranspose: **Kaiming Normal** with `a=0.2` (matched to LeakyReLU slope), `mode='fan_out'`.
- BatchNorm: weight = 1, bias = 0 (standard).

### 4.8 Parameter Count

| Configuration | Parameters |
|---|---|
| `base_channels=16` (light) | ~263 K |
| `base_channels=32` (standard) | ~1.05 M |
| `base_channels=64` (production) | ~4.19 M |

Production model (`base_channels=64`, `depth=4`): **4,186,242 parameters**.

---

## 5. COMPLEX PHASE-AWARE LOSS FUNCTION

### 5.1 Motivation

A naive MSE loss on raw IQ channels \(\mathcal{L}_{MSE} = \mathbb{E}[(I_{pred}-I_{tgt})^2 + (Q_{pred}-Q_{tgt})^2]\) is mathematically insufficient for this problem because:

1. A model that outputs zero everywhere for silence samples achieves good IQ MSE but completely fails to reconstruct pulse positions (zero-amplitude output would pass every silent sample with perfect MSE).
2. A pulse shifted by one sample (0.5 µs) causes a CRC failure but has small IQ MSE if the amplitude is correct.
3. Phase errors are non-trivially penalised by IQ MSE — a 90° phase rotation at full amplitude produces an IQ MSE of 2.0 (same as missing the pulse entirely), even though the physical error is merely a carrier phase offset which does not affect PPM decoding.

### 5.2 Composite Loss Definition

$$\mathcal{L}_{total} = w_{iq} \cdot \mathcal{L}_{iq} + w_{mag} \cdot \mathcal{L}_{mag} + w_{phase} \cdot \mathcal{L}_{phase}$$

**Default weights (synthetic training):** \(w_{iq}=1.0,\ w_{mag}=0.5,\ w_{phase}=0.3\)

**Supervised training weights:** \(w_{iq}=1.0,\ w_{mag}=0.5,\ w_{phase}=0.5\) (higher phase weight — perfect re-synthesized targets make phase alignment more reliable)

#### Component 1: Channel MSE

$$\mathcal{L}_{iq} = \frac{1}{B \cdot 2 \cdot N} \sum_{b,c,n} \left(X^{pred}_{b,c,n} - X^{tgt}_{b,c,n}\right)^2$$

This is the **workhorse loss** — directly drives reconstruction of exact IQ waveforms including all pulse positions. Implemented as `torch.nn.functional.mse_loss(pred, target)` applied to the full `(B, 2, N)` tensor.

#### Component 2: Magnitude Envelope MSE

$$\mathcal{L}_{mag} = \frac{1}{B \cdot N} \sum_{b,n} \left(\sqrt{I^{pred}_{b,n}{}^2 + Q^{pred}_{b,n}{}^2 + \varepsilon} - \sqrt{I^{tgt}_{b,n}{}^2 + Q^{tgt}_{b,n}{}^2 + \varepsilon}\right)^2$$

where \(\varepsilon = 10^{-7}\) prevents gradient explosion at the origin.

This term forces the model to correctly reconstruct which samples are "on" (magnitude ≈ amplitude) and "off" (magnitude ≈ 0). Without this, a model could achieve zero \(\mathcal{L}_{iq}\) by outputting a rotated IQ constellation where all pulses are present but offset in phase — correct power but wrong bit positions.

#### Component 3: Circular Phase Error (The Wrap-Around Safe Formulation)

$$\mathcal{L}_{phase} = \mathbb{E}\left[(1 - \cos\Delta\phi) \cdot w_{signal}\right]$$

where:
$$\Delta\phi_n = \text{atan2}(Q^{pred}_n, I^{pred}_n) - \text{atan2}(Q^{tgt}_n, I^{tgt}_n)$$

**Why \((1 - \cos\Delta\phi)\) and not \(|\Delta\phi|\):**

The raw phase difference \(\Delta\phi = \phi_{pred} - \phi_{tgt}\) suffers from **wrap-around discontinuity** at \(\pm\pi\). For example, \(\phi_{pred} = +3.1\) and \(\phi_{tgt} = -3.1\) rad represents a true angular difference of only 0.08 rad, but naive subtraction gives \(\Delta\phi = 6.2\) rad — a factor of 78× overestimate of the true error.

The \((1 - \cos\Delta\phi)\) formulation avoids this:
- \(\Delta\phi = 0\): \(\cos(0) = 1\) → error = **0** (perfect)
- \(\Delta\phi = \pi/2\): \(\cos(\pi/2) = 0\) → error = **1**
- \(\Delta\phi = \pi\): \(\cos(\pi) = -1\) → error = **2** (maximum)
- **Wrap-safe:** \(\cos(\Delta\phi) = \cos(\Delta\phi + 2\pi k)\) for any integer \(k\)

This is equivalent to the **squared chord distance** on the unit circle.

#### The Amplitude Mask: \(w_{signal}\)

Phase is geometrically undefined at the origin (where \(I = Q = 0\)). During silence periods of the OOK carrier, both target channels are zero — `atan2(0, 0)` is undefined and penalising it would add noise to the gradient. The mask:

$$w_{signal}(n) = \frac{|IQ^{tgt}_n|}{\text{mean}(|IQ^{tgt}|) + \varepsilon}$$

weights the phase penalty proportionally to the target magnitude. During silence gaps, \(|IQ^{tgt}| \approx 0\), so \(w_{signal} \approx 0\) and the phase term contributes nothing. During pulse-on samples, \(w_{signal} \approx 1\) (normalised by the batch mean), and phase fidelity is fully enforced. Dividing by the batch mean ensures the scale of \(\mathcal{L}_{phase}\) is dataset-independent.

---

## 6. RESULTS & MILESTONES ACHIEVED

### 6.1 Phase 1: Synthetic Pre-Training

**Configuration:**
- Model: `IQAutoencoder(base_channels=64, depth=4)` — 4,186,242 parameters
- Dataset: `SyntheticADSBDataset` — 8,192 train / 1,024 val samples
- Collision fraction: 25%
- SNR range: 8–25 dB
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-5)
- Loss weights: (1.0, 0.5, 0.3) — (w_iq, w_mag, w_phase)
- Device: Apple Silicon MPS

**After 5 Epochs (rapid convergence test):**
- Initial total loss: ~3.55 → after 5 epochs: ~0.096
- Loss reduction ratio: **~37×**
- Interpretation: The model rapidly locates the high-gradient geometry of the problem (pulse on/off structure is learned in the first few epochs).

**After 50 Epochs (full synthetic run):**
- Train IQ MSE component descended from ~2.38 to ~2.18 (normalised scale)
- Magnitude MSE: 1.06 → 0.82 — **beating envelope flattened**
- Phase component: 0.947 → 0.825 — circular phase fidelity improved
- Visual results from `evaluate.py`:
  - Constellation plots: noisy Lissajous figure → clean circle (primary aircraft only)
  - Magnitude envelope: 42 kHz beating oscillation (amplitude peak 1.75×) suppressed back to unit circle (flat 1.0)
  - Mean per-sample IQ MSE: 0.3557 (input) → 0.2341 (reconstructed) — **34.2% reduction**
- Checkpoint saved: `checkpoints/best.pt`

### 6.2 Phase 2: Supervised Real-World Fine-Tuning

**Dataset extraction (`extract_labels.py`):**
- Source: 91 real RTL-SDR `.npy` burst files from `adsb_iq_sample_collection`
- Antennas: `rpi_east` (March 11 + April 15, 2026), `rpi_west` (March 16 evening)
- Valid CRC-passing frames extracted: **7,357 supervised pairs**
- Extraction time: 38.5 seconds for 91 × 5-second bursts
- Average frames per burst: ~81
- Preamble detector: adaptive threshold `thr = max(noise_floor × 3.5, 0.10)` with shape test `high > low × 3.0`
- Clean target generation: re-synthesised via `generator.py` using `SignalParams` estimated from real preamble (amplitude, phase \(\phi_0\), and frequency offset \(f_{offset}\) estimated from preamble pulse phases)

**Frequency offset estimation from real preamble:**

$$\hat{f}_{offset} = \frac{1}{2} \left(\frac{\Delta\phi_{02}}{2 \cdot 2\pi \cdot T_s} + \frac{\Delta\phi_{79}}{2 \cdot 2\pi \cdot T_s}\right)$$

where \(\Delta\phi_{02} = \arg(e^{j(\phi_2 - \phi_0)})\) and \(\Delta\phi_{79} = \arg(e^{j(\phi_9 - \phi_7)})\) (wrap-safe complex argument).

**Fine-tuning configuration:**
- Warm start: `checkpoints/best.pt` (104/104 tensors transferred — perfect architecture match at `base_channels=64`)
- Encoder frozen for first 5 epochs (decoder-only adaptation phase)
- Optimizer: Adam, lr=1e-4 (10× lower than pre-training — prevents catastrophic forgetting)
- Epochs: 50
- Batch size: 64
- Loss weights: (1.0, 0.5, 0.5) — increased phase weight
- Augmentation: phase rotation ∈ [0, 2π), amplitude jitter ±10%, per-window RMS normalisation
- Train/val split: 6,254 / 1,103 (85/15)
- Device: Apple Silicon MPS

**Convergence:**
- Epoch 1: train total = 3.391, val = 3.342
- Epoch 50: train total = 3.006, val = 3.121
- Best val loss: **3.065** at epoch 41
- 487 unique ICAO addresses in training set (good aircraft diversity)

### 6.3 Phase 3: Real-World Evaluation

**Evaluation methodology (`compare_decodings.py`):**
- Input file: `adsb_capture.bin` — 480 MB, 240M samples, real RTL-SDR capture
- Processing: `live_bridge.py` with sliding window (hop=120, batch=32, 50% overlap, Hanning OLA synthesis)
- Per-window RMS normalisation auto-detected from checkpoint `training_mode` field

**Results:**

| Condition | Noise floor | Preamble candidates | Valid frames | Unique aircraft |
|---|---|---|---|---|
| Raw signal | 0.110 | 25,371 | **1,589** | **31** |
| `best.pt` (synthetic only) | 0.507 (4.6× higher) | 3 | 0 | 0 |
| `best_supervised.pt` (fine-tuned) | **0.069 (0.63×, lower than raw)** | 16,097 | **116** | **17** |

**Key observation:** The supervised fine-tuned model is **suppressing noise below the raw hardware floor** (noise floor 0.069 vs. raw 0.110), demonstrating genuine denoising. The 116 CRC-valid frames recovered from the processed stream (vs. 0 from the synthetic-only model) validate the domain-adaptation approach. Recovery rate: 7.3% of the raw stream's frames — establishing a non-zero baseline from which further training will improve.

### 6.4 Phase 4: Scaled Training — v2 with Collision Augmentation

**Dataset:** 152,343 supervised pairs extracted from 406 GB of real captures (via MinIO/SSD pipeline, §8.1). 785 label files, 2,016 unique ICAO addresses — a 4× improvement in aircraft diversity over the v1 dataset.

**Training configuration:**
- Warm start: `best_supervised.pt` (v1) — 104/104 tensors transferred
- Collision augmentation: 30% probability (`--collision-augment 0.30`)
- Synthetic interferer: **clean** (no AWGN added — identified post-hoc as a design flaw, see §6.5)
- Epochs: 50 (no early stopping — model kept improving throughout)
- Device: Apple Silicon MPS, ~83 s/epoch, ~69 min total

**Convergence (best val losses per checkpoint):**

| Epoch | Val Loss | Notes |
|---|---|---|
| 1 | 2.781 | Warm-start baseline |
| 14 | 2.359 | First significant best |
| 30 | 2.255 | Still improving, no plateau |
| 48 | **2.208** | Final best (saved to `best_supervised_v2.pt`) |

**Loss component breakdown at epoch 48 (best):**
- IQ MSE: 1.554
- Magnitude: 0.622 (37% lower than epoch 1 — significantly sharper PPM pulse envelopes)
- Phase: 0.686

### 6.5 Phase 4 Post-Evaluation: v2 Regression Discovery

**Real-world evaluation on `adsb_capture.bin` (480 MB, same test file as §6.3):**

| Model | Noise Floor | Preamble Candidates | Valid Frames | Aircraft |
|---|---|---|---|---|
| Raw signal | 0.1101 | 25,371 | **1,589** | **31** |
| `best_supervised.pt` (v1) | 0.0687 (−38%) | 16,097 | **116** | **17** |
| `best_supervised_v2.pt` (v2) | 0.0517 (−53%) | 24,329 | **8** | **8** |

**Result:** v2 shows stronger denoising (noise floor fell further to 0.0517) but a catastrophic regression in valid frame recovery: 8 frames vs. v1's 116 frames. This is worse than the v1 result despite the superior training loss.

**Root cause analysis — "Clean Interferer" over-suppression:**

The v2 model's 30% collision augmentation used a **perfectly noise-free synthetic interferer** (signal B from `generator.py` with no AWGN). Over 50 epochs and 152K pairs, the model learned:

> *"If an input contains a clean, geometrically perfect ADS-B-structured signal, suppress it."*

This rule is correct for the synthetic interferer. But at inference time, the **primary aircraft signal** — after per-window RMS normalisation — also appears as a relatively clean, structured pattern. The model partially suppresses it alongside the intended interferer, distorting PPM bit boundaries enough to cause CRC failures. The 24,329 preamble candidates in the v2 output confirm that the model is not simply outputting noise — it is producing structurally plausible but subtly corrupted waveforms.

The 30% augmentation rate compounded this effect: roughly 1 in 3 training examples trained the model to suppress, vs. 2 in 3 training it to denoise-only. Over 50 epochs × 152K pairs, the suppression behaviour became the dominant learned response.

**This is now documented as a critical design constraint** (§7.7).

### 6.6 Pipeline Validation — Infrastructure vs. Model Failure (July 2026)

Before attributing zero-frame results to domain shift or collision training gaps, the `live_bridge.py` inference pipeline was validated in isolation on `test_capture_36db.bin` (120 MB, 30 s, 1090 MHz centre, 36 dB gain, 314 CRC-valid frames raw).

#### 6.6.1 DC Removal Mismatch (Fixed)

**Problem:** Per-window DC subtraction (240 samples) inflated the window mean when an ADS-B pulse was present inside the window, over-removing DC and distorting pulse shape. Training data (`extract_labels.py`) subtracts the **full-burst mean** (~60 M samples), where pulse energy is negligible relative to noise.

**Fix:** `live_bridge.py` now accumulates the first ~500 K raw samples during a DC warmup phase, computes a global `dc_i` / `dc_q` estimate, freezes it, and subtracts that constant from every inference window thereafter.

**Outcome on `test_capture_36db.bin`:** Warmup completed with `dc_i = −0.001043`, `dc_q = −0.001003` — DC bias is ~0.1% of full scale on this capture. Fixing DC removal did **not** change the zero-frame result; the hardware DC offset was never the dominant failure mode on this file.

#### 6.6.2 Identity-Mode End-to-End Test (`--identity`)

A new `--identity` CLI flag runs the **full bridge** (OLA, DC warmup, per-window RMS norm/denorm) but replaces the model with `output = input`. This distinguishes pipeline bugs from model behaviour.

```bash
python live_bridge.py \
    --ckpt checkpoints/best_supervised.pt \
    --identity \
    --input test_capture_36db.bin \
    --output test_identity_pipeline.bin

python compare_decodings.py test_capture_36db.bin test_identity_pipeline.bin
```

**Results:**

| Condition | Valid frames | Unique aircraft | Preamble candidates |
|---|---|---|---|
| Raw `test_capture_36db.bin` | **314** | **9** | 11,543 |
| Identity pipeline (`--identity`) | **314** | **9** | 11,545 |
| `best_supervised.pt` (v1, real model) | **0** | **0** | ~559 |

**Conclusion:** OLA synthesis, DC warmup, windowing, and RMS normalisation round-trip **preserve all decodable frames**. The zero-frame failure is entirely attributable to **learned model weights**, not inference infrastructure.

**Contrast with `--passthrough`:** Passthrough mode skips the bridge entirely (uint8→float→uint8 per chunk). It validates IQ conversion rounding only; it does **not** exercise OLA or windowing.

### 6.7 `test_capture_36db.bin` — Capture-Specific Model Failure (July 2026)

All three supervised checkpoints were evaluated on `test_capture_36db.bin` after the DC warmup fix:

| Model | Valid frames | Notes |
|---|---|---|
| Raw input | **314** | Baseline — capture decodes well without processing |
| `best_supervised.pt` (v1) | **0** | Previously achieved 116 frames on `adsb_capture.bin` (§6.3) |
| `best_supervised_v2.pt` (v2) | **0** | Over-suppression regression documented in §6.5 |
| `best_supervised_v3.pt` (v3) | **0** | v3 fixes (10% collision, noisy interferer) did not recover this capture |

**Key observations:**

- The model output retains structure (hundreds of preamble candidates) but **fails CRC** — it produces plausible-looking but subtly corrupted waveforms, not white noise.
- This capture sits **outside the training distribution** relative to the 406 GB corpus (different gain, site, or capture conditions). Supervised fine-tuning is not universally transferable across all real captures.
- Problem **A** (single-aircraft denoising failure) must be resolved before Problem **B** (collision/BSS) is relevant on this file — the raw capture contains no collision test scenario; all 314 frames are ordinary single-aircraft decodes.

**v3 status:** Training completed (`best_supervised_v3.pt`, val loss improved over v1). Real-world evaluation on `test_capture_36db.bin` did not recover frames. Evaluation on the original `adsb_capture.bin` benchmark (§6.3) is pending to determine whether v3 exceeds v1's 116-frame baseline on in-distribution data.

### 6.8 v4 Denoising-Only Retrain (July 2026)

**Goal:** Phase 1 of revised training plan — collision augment disabled, warm-start from v1, 25 epochs.

**Configuration:**
- Warm start: `best_supervised.pt` (v1)
- `--collision-augment 0.0`
- 152,343 pairs, 25 epochs, best val loss **2.303** (epoch 25) — much lower than v1's ~3.06
- Checkpoint: `checkpoints/best_supervised_v4.pt`

**Holdout evaluation (decode-gated — the metric that matters):**

| Capture | Raw frames | v1 | v4 | v4 recovery |
|---|---|---|---|---|
| `adsb_capture.bin` (in-distribution) | 1,589 | **116** | **83** | 5.2% (regression vs v1) |
| `test_capture_36db.bin` (OOD) | 314 | 0 | **1** | 0.3% (marginal; still failed gate) |

**Observations:**
- Lower val loss again **anti-correlates** with frame recovery (same pattern as v2).
- v4 on `adsb_capture.bin`: noise floor 0.110 → 0.059, preamble candidates 25,371 → 67,166 — aggressive processing with corrupted CRC structure.
- v4 on `test_capture_36db.bin`: 1 frame vs 0 for v1/v2/v3 — not a meaningful pass; gate requires ≫0 on OOD capture.
- **Phase 1 gate: failed.** Next step: blend sweep (`blend_sweep.py`) to find usable α, then v4b with target blending and decode-gated checkpoint selection.

**Blend diagnostic (offline `blend_sweep.py` and online `--blend 0.05`, July 2026 — validated):**

| Capture | Raw frames | v4 α=1 | v4 α=0.05 online `--blend` | Recovery |
|---|---|---|---|---|
| `adsb_capture.bin` | 1,589 | 83 | **1,589** | 100% |
| `test_capture_36db.bin` | 314 | 1 | **295** | 93.9% |

Online `--blend` matches offline `blend_sweep.py` after emit-buffer fix (partial raw coverage, no sample loss). Deployable artifacts: `deploy_adsb_capture_blend.bin`, `deploy_test_capture_blend005.bin`.

```bash
python live_bridge.py --ckpt checkpoints/best_supervised_v4.pt \\
    --blend 0.05 --input adsb_capture.bin --output clean.bin
```

---

## 7. KNOWN LIMITATIONS & EDGE COGNIZANCE

### 7.1 The Total Phase Cancellation Problem

When two signals arrive with equal amplitude and exactly 180° phase difference at a pulse position:

$$r(t) = A \cdot e^{j\phi} + A \cdot e^{j(\phi + \pi)} = A \cdot e^{j\phi}(1 + e^{j\pi}) = A \cdot e^{j\phi} \cdot 0 = 0$$

The received sample at that position is identically zero — the two signals cancel completely. In a magnitude-only decoder this is indistinguishable from silence. The IQ envelope shows zero energy.

The model has no instantaneous information at these samples, but it can use **structural memory** across adjacent samples: the phase arc trajectory established by the samples immediately preceding the cancellation point, and the preamble phase fingerprint captured by the bottleneck's global frame context, allow it to estimate the likely pulse position and reconstruct it. However, if multiple consecutive samples cancel, reconstruction becomes unreliable. This is an irreducible information-theoretic limit — no receiver can fully recover a signal from a window where it has zero energy contribution.

**Probability of full cancellation:** For uniformly distributed phase difference, \(P(|\Delta\phi - \pi| < \delta) = \delta/\pi\). With typical oscillator drift this is low per-sample but non-negligible over 112-bit frames.

### 7.2 Domain Shift (Synthetic → Real)

The model pre-trained purely on synthetic data (Phase 1) produced **0 valid frames** on real captures. Root cause analysis:

- Real RTL-SDR noise is not perfectly Gaussian — it includes 1/f flicker noise, intermodulation products, and interference from co-located electronics.
- Real DC offset has a complex spectral shape (not a pure DC spike) after conversion from IF.
- Real signals have amplitude fading from multipath reflection (building reflections, aircraft fuselage geometry) not present in the synthetic model.
- The synthetic training data used fixed noise statistics; real hardware output varies with temperature (heat wave data is particularly diverse).

This mandated the supervised fine-tuning approach (Phase 2), which resolved the zero-frame result to 116 frames on `adsb_capture.bin` by training on decoded-and-re-synthesized real captures — but **does not guarantee transfer to all captures**. An out-of-distribution file (`test_capture_36db.bin`, 1090 MHz, 36 dB gain) yields 0 valid frames from v1, v2, and v3 despite 314 raw decodes (§6.7, §7.8.4).

### 7.3 Sparse Supervised Labels and Absence of Real Collision Examples

Only ~7.3% of real burst windows contain valid CRC-passing frames — the remaining 92.7% is inter-frame noise. The supervised dataset (`extract_labels.py`) only trains on the valid windows. This creates:

- **Data scarcity:** 7,357 pairs from 7 GB of captures (vs. theoretical maximum if all windows could be labelled).
- **Positive-only bias:** The model never sees real noise-only windows during supervised training (only during synthetic pre-training). This may cause it to be overconfident about reconstructing signals from pure noise windows at inference.
- **No real collision examples:** A co-channel collision almost always corrupts the CRC. `extract_labels.py` only extracts CRC-passing frames, so every supervised pair is a **single-aircraft** example. Collision resolution capability comes entirely from the 25% synthetic collision fraction during Phase 1 pre-training, and does not improve from real data collection alone.

**Mitigation for noise-only windows:** The sliding-window OLA inference in `live_bridge.py` averages responses over overlapping windows, attenuating hallucinated signals from noise-only input.

**Mitigation for absent collision examples:** Synthetic collision augmentation (§3.4) applied on-the-fly during training.

### 7.4 Edge Hardware Performance

| Platform | Device | Throughput | Real-time ratio | Notes |
|---|---|---|---|---|
| MacBook M-series | MPS | ~1.07 MB/s | 0.27× | Bottleneck: MPS launch overhead at batch_size=32 |
| Raspberry Pi 4 | CPU (4 threads) | ~0.02 MB/s* | ~0.005× | *Sandbox-throttled measurement; expected ~0.1–0.3× in production |
| Raspberry Pi 5 | CPU | ~2–3× RPi4 | TBD | |
| Intel N100 (x86) | CPU | ~5–10× RPi4 | TBD | AVX2 optimised PyTorch |

The 0.27× real-time ratio on MPS means the model processes 2 MSPS data at ~540,000 samples/second — suitable for post-capture processing but not live streaming at this batch size. Larger batch sizes or TorchScript compilation (`--torchscript` flag in `live_bridge.py`) improve throughput.

**Mitigation options (ranked by implementation complexity):**
1. Increase `--hop` to 240 (no overlap) — 2× throughput, slight edge artefact risk.
2. Increase `--batch-size` to 128 — amortises MPS/CPU launch overhead.
3. `--torchscript` flag — traces the model with `torch.jit.trace`, eliminating Python dispatch overhead.
4. Reduce `base_channels` to 32 for edge deployment (~1.05M params, ~4× faster).
5. INT8 quantisation (future work).

### 7.5 Fixed Intermediate Frequency (IF) Offset Constraint

#### 7.5.1 The Constraint

The current `IQAutoencoder` is strictly optimised for a **single, fixed Intermediate Frequency offset** — the hardware IF at which all training data was captured. In the deployed system, the RTL-SDR is tuned 250 kHz above 1090 MHz (`-f 1090250000`), placing the ADS-B passband at a nominal +250 kHz IF in the complex baseband representation. Every aircraft signal therefore arrives as a phasor rotating at a base angular velocity of:

$$\omega_{IF} = 2\pi \cdot f_{IF} \cdot T_s = 2\pi \times 250{,}000 \times 0.5 \times 10^{-6} \approx 0.785 \text{ rad/sample}$$

plus each aircraft's individual oscillator offset \(\delta f_i \in [-50, +50]\) kHz, yielding total angular velocities in the range:

$$\omega_{total} \in [2\pi \cdot 200{,}000 \cdot T_s,\ 2\pi \cdot 300{,}000 \cdot T_s] \approx [0.628,\ 0.942] \text{ rad/sample}$$

All synthetic training data in `generator.py` samples `f_offset` uniformly from this window (`f_offset_range = (-50_000.0, 50_000.0)` Hz centred on the 250 kHz IF). All real-world supervised labels in `extract_labels.py` were captured at the same hardware configuration. The model has therefore **never been exposed to signals rotating outside this angular velocity band**.

#### 7.5.2 Why the Architecture Encodes IF as a Geometric Prior

The 1D-CNN encoder learns its filters through gradient descent on the actual IQ trajectories present in training data. At a given IF offset, the on-state samples of a valid ADS-B pulse trace a specific arc on the complex plane: the angular displacement between the sample at the start of a pulse and the sample at its end is exactly \(\omega_{total} \cdot \Delta n\) radians. The encoder's first-stage filters (`Conv1d(2, 64, kernel=3)`) develop **matched filter banks** that are tuned to detect these specific arc lengths and curvatures.

Critically, **the rate of phase rotation per sample is baked into every learned filter weight.** A filter tuned to recognise a 0.785 rad/sample arc will produce a strong response to pulses from a 250 kHz IF signal, and a weak (or adversarial) response to pulses from a signal rotating at 0.2 rad/sample (a 63 kHz IF, e.g. tuned directly to 1090 MHz with zero offset). This is not a bug — it is a consequence of using a spatially-tied convolutional kernel to detect a physically characterised trajectory. The network is, in effect, a **learned matched filter bank for one specific phase velocity regime**.

This extends to the bottleneck's global context integration. The beating envelope (the BSS discriminant) has a spatial frequency in the latent time axis that is directly proportional to \(|f_a - f_b|\) expressed in **angular distance per latent step** — which is itself a function of the IF. Changing the IF shifts the entire angular velocity distribution of all signals in the training set, making the beating envelope appear at a different spatial frequency in the bottleneck. The bottleneck filters are not invariant to this shift.

#### 7.5.3 Consequences of Changing the IF Offset

If the hardware is reconfigured to tune directly to 1090 MHz (zero IF) or to a different offset (e.g., +100 kHz or +500 kHz), the following failures occur:

| IF Change | Effect on model |
|---|---|
| Zero-IF (DC-coupled, 0 Hz) | Signals rotate at ~0 rad/sample near DC; encoder outputs near-zero activations; model outputs near-zero (complete reconstruction failure) |
| Different fixed offset (e.g., +100 kHz) | Arc length per pulse mismatches all learned filter responses; loss function may appear to converge but CRC failure rate remains high |
| Opposite sign (e.g., −250 kHz) | Complex conjugate of trained trajectory; model sees mirror-image arcs; produces incorrect phase reconstruction |

The model does **not** learn to handle variable IF — it learns to perfectly handle one IF. This is the fundamental trade-off of the current architecture: **specialisation for maximum performance on a fixed, known hardware configuration**, rather than general-purpose SDR-agnostic operation.

#### 7.5.4 Domain Randomisation: Deferred to Future Work

A generalised model capable of handling arbitrary IF offsets would require **domain randomisation** at the data generation level: uniformly sampling the centre IF offset across the full RTL-SDR passband (e.g., \(f_{IF} \in [-1\,000\,000, +1\,000\,000]\) Hz) during both synthetic pre-training and supervised fine-tuning. Each training sample would present the model with ADS-B pulses rotating at a completely different angular velocity, forcing the encoder to learn **IF-invariant features** — pulse shape relative to background, magnitude envelope transitions, and the ratio of on/off sample magnitudes — rather than absolute arc curvatures at a specific angular rate.

This generalisation is explicitly deferred because:

1. **Model capacity cost.** IF-invariant representation requires significantly more filter diversity in the encoder's first stage. At the minimum, the channel count at the first encoder layer would need to span the full angular velocity range (roughly 4–5× the current range), increasing `base_channels` and therefore parameter count and inference latency on edge hardware.

2. **Computational cost at training time.** Domain randomisation expands the effective training distribution by ~20× (1 MHz / 50 kHz = 20 distinct IF sub-ranges), requiring proportionally more training samples and epochs for the encoder to adequately cover the space.

3. **Unnecessary for the deployed system.** The hardware configuration (RTL-SDR tuned to +250 kHz IF, `rtl_sdr -f 1090250000`) is **fixed** across all deployed units (`rpi_east`, `rpi_west`, and any future nodes). There is no operational requirement for IF flexibility — the tuning offset is a deployment constant, not a variable. Retraining on a new IF takes one data-collection cycle.

4. **Performance optimality of specialisation.** A model trained on exactly the IF it will encounter at inference can devote all of its representational capacity to the fine-grained features that discriminate between valid pulses and noise at that specific angular velocity — achieving higher denoising quality than a general model of equal parameter count.

**Deferred implementation path (when IF flexibility is needed):**
```python
# In generator.py generate_batch() — future domain randomisation extension:
# Sample centre IF uniformly; per-sample f_offset is then IF_centre + aircraft_drift
IF_centre = self.rng.uniform(-500_000.0, 500_000.0)  # Hz
f_offset = IF_centre + self.rng.uniform(-50_000.0, 50_000.0)
```

Combined with a metadata-conditioned decoder (feeding \(f_{IF}\) as a scalar embedding to each decoder block), this would allow a single model to serve arbitrary RTL-SDR hardware configurations without retraining.

### 7.6 Collector OOM Failure and Mitigation

#### 7.6.1 Incident Summary

After several hours of continuous operation, both `rpi_east` and `rpi_west` collector services were terminated by the Linux OOM killer. Collection stopped after accumulating **406 GB** on the central SSD. No data was lost from files already uploaded to MinIO; only the in-progress burst at the time of termination on each Pi was discarded.

#### 7.6.2 Root Cause Analysis

Two compounding memory issues were identified in `ml_data_collector.py`:

**Issue 1 — Implicit complex128 intermediate allocation.** The original conversion expression:
```python
iq_mm[sample_idx:end_idx] = (I + 1j * Q).astype(np.complex64)
```
The Python literal `1j` is `complex` (128-bit). NumPy promotes `1j * Q` (where Q is `float32`) to `complex128` before the `.astype(np.complex64)` cast. This creates two temporary 128-bit arrays (`1j * Q` and `I + 1j * Q`) per chunk before the final 64-bit result is written. For a 262,144-byte chunk, the peak temporary allocation is ~5× larger than necessary. Across hundreds of iterations with Python's non-returning allocator, this creates heap fragmentation that slowly grows RSS over the service lifetime.

**Issue 2 — No process lifetime limit.** The main loop ran indefinitely without a clean restart point. Python's memory allocator (`pymalloc`) returns freed blocks to its own pool but does not always return pool pages to the OS. Over hundreds of burst cycles, each involving a 480 MB memmap, a MinIO upload read buffer, and multiple temporary numpy arrays, the resident set size grows monotonically regardless of actual live object count.

#### 7.6.3 Applied Fixes (in `ml_data_collector.py`)

1. **Eliminated complex128 intermediate** — replaced with explicit `np.empty(n_iq, dtype=np.complex64)` with in-place `.real` / `.imag` assignment, plus explicit `del chunk_iq, I, Q, buf` to release temporaries within the loop immediately:
   ```python
   chunk_iq = np.empty(n_iq, dtype=np.complex64)
   chunk_iq.real[:] = I
   chunk_iq.imag[:] = Q
   iq_mm[sample_idx:end_idx] = chunk_iq
   del chunk_iq, I, Q, buf
   ```

2. **Periodic forced GC** — `gc.collect()` called after every burst to reclaim any reference-cycle garbage before it accumulates.

3. **`--max-bursts` self-restart mechanism** — the process now exits cleanly after `N` bursts (default: 60 bursts = 30 minutes at 30 s/burst). `systemd`'s `Restart=always` policy immediately relaunches the process with a fresh heap. This is zero-downtime for data collection (systemd restart latency < 2 seconds) and completely eliminates long-term RSS growth:

   ```
   Burst 1–60   → service exits cleanly
   ↓ systemd restarts (< 2 s gap)
   Burst 61–120 → service exits cleanly
   ...
   ```

4. **`--max-bursts 0`** disables the mechanism for environments with larger RAM (e.g., RPi 4 with 8 GB) where OOM is not a concern.

### 7.7 Collision Augmentation Over-Suppression (v2 Regression)

#### 7.7.1 The Problem

When a synthetic interferer is added with a high probability (≥30%) and the interferer is a **perfectly noise-free** synthesized signal, the model learns two distinct behaviours:

1. **Denoising** (from the 70% clean-input examples): map noisy real IQ → clean re-synthesized IQ.
2. **Suppression** (from the 30% collision examples): detect and remove clean structured patterns.

These two objectives are in direct tension. At inference time, both the denoising target *and* the suppression target are "clean structured patterns" — the model cannot distinguish a synthetic interferer from the primary aircraft signal under its learned feature representation, particularly after RMS normalisation removes the amplitude discriminant.

#### 7.7.2 The Fix (v3 Training)

Two complementary fixes were applied:

**Fix 1 — Reduce collision augmentation rate to 10%:**
At 10% augmentation, the model sees ~9 clean denoising examples for every 1 collision example. This ensures the denoising objective dominates the gradient signal, limiting how aggressively the suppression response can develop.

**Fix 2 — Add AWGN to the synthetic interferer (`interferer_snr_db`):**
Before superimposing signal B onto the real noisy input, Gaussian noise is added to signal B at a configurable SNR (default 20 dB). This makes the synthetic interferer indistinguishable from a real distant aircraft in terms of its noise statistics, preventing the model from exploiting the "no noise = synthetic, suppress it" shortcut.

```python
# From supervised_dataset.py _add_synthetic_interferer() — v3 fix:
if self.interferer_snr_db is not None:
    sig_power = float(np.mean(clean_b ** 2)) + 1e-12
    noise_var = sig_power / (10.0 ** (self.interferer_snr_db / 10.0))
    clean_b = clean_b + rng.normal(0.0, float(np.sqrt(noise_var)), size=clean_b.shape)
```

At 20 dB SNR, the noise on signal B is ~10% of its RMS amplitude — perceptible but not signal-distorting. The beating envelope (the primary BSS discriminant) remains fully intact.

#### 7.7.3 v3 Training Command

```bash
python train_supervised.py \
    --labels-dir "/Volumes/KBMA SSD/labels" \
    --warm-start  checkpoints/best_supervised.pt \
    --ckpt-out    checkpoints/best_supervised_v3.pt \
    --epochs 50 \
    --collision-augment 0.10 \
    --interferer-snr-db 20.0
```

Key differences from v2: warm-start from **v1** (not v2), collision rate 10% (not 30%), noisy interferer (not clean).

#### 7.7.4 General Design Constraint

For future collision augmentation work, the following rule applies:

> **Never use a perfectly clean synthetic signal as a collision interferer in a training setup where the clean target is also a structured synthetic signal.** The model will learn to suppress structural perfection, which is the opposite of the denoising objective.

The synthetic interferer must be at least as noisy as the primary aircraft signal's noise floor to prevent this shortcut. For the current training setup, `interferer_snr_db = 20` dB (matching a typical medium-range aircraft SNR) is the recommended floor.

### 7.8 Current Constraints Summary — What Is Actually Blocking Progress (July 2026)

Pipeline validation (§6.6) established that inference infrastructure is sound. The remaining constraints are **data, objective, and distribution** — not OLA, DC, or uint8 conversion. They are listed below in priority order.

#### 7.8.1 No True Clean RF Reference

The project never observes `(noisy_real, clean_real)` from the same propagation path. Available training proxies:

| Approach | Input | Target | Gap |
|---|---|---|---|
| `extract_labels.py` + `train_supervised.py` | Real noisy window | Re-synthesized perfect frame from decoded bits | Target is mathematically ideal, not what the antenna received (multipath, pulse shape, phase trajectory differ) |
| `train_real.py` (Noise2Signal) | Real window + added Gaussian | Same real window (DC removed) | Assumes real noise ≈ Gaussian; "remove added noise" may not generalize to suppressing native RTL-SDR noise |
| Collision augment (`supervised_dataset.py`) | Real noisy A + synthetic B | Re-synth of A only | B's geometry may be correct; B's noise texture, pulse edges, and phase evolution remain synthetic |

Adding Gaussian noise to a synthetic interferer (v3 fix, §7.7.2) prevents the "suppress clean structure" shortcut but does **not** make B statistically identical to a real second aircraft.

#### 7.8.2 No Real Collision Ground Truth

Real co-channel collisions almost always fail CRC → `extract_labels.py` silently discards them. Every supervised pair is **single-aircraft**. BSS/collision capability can only come from:

- Phase 1 synthetic pre-training (~25% collision fraction in `generate_batch()`), and/or
- On-the-fly synthetic collision augmentation on real single-aircraft windows

Neither path provides labels where the model must separate **two real RF sources** with independent noise, multipath, and oscillator drift. Phase inconsistency and beating in real collisions cannot be fully learned from synthetic geometry alone.

#### 7.8.3 Training Objective ≠ Decoder Success Metric

The model minimizes IQ / magnitude / phase reconstruction loss against a synthetic clean target. Downstream success requires **PPM bit edges in the magnitude envelope** to survive CRC validation. These objectives diverge:

- **v2 proved this explicitly:** val loss 2.208 (best ever) but only 8 valid frames vs. v1's 116 (§6.5).
- Lower noise floor at inference does not imply better frame recovery — the model can suppress noise while simultaneously smoothing pulse boundaries.

Collision augmentation compounds this: a fraction of training teaches **suppression** (remove structured signal B), while the majority teaches **denoising** (map noisy → clean). After RMS normalisation, the primary aircraft also appears as a clean structured pattern — the model cannot reliably distinguish "target to preserve" from "interferer to remove" under the current loss.

#### 7.8.4 Capture-Specific Domain Shift

Supervised fine-tuning is not binary (works / fails on all real data):

- **Synthetic-only (`best.pt`):** 0 frames on real captures.
- **v1 (`best_supervised.pt`):** 116 frames on `adsb_capture.bin` (§6.3) — partial success.
- **v1 / v2 / v3 on `test_capture_36db.bin`:** 0 frames each (§6.7) — out-of-distribution relative to training.

Real RTL-SDR noise includes 1/f flicker, IQ imbalance, intermodulation, gain-dependent effects, and multipath fading (§7.2). Gaussian noise injection and synthetic collision augment do not close this gap if the **base capture statistics** (gain, site, antenna, temperature) differ from the 406 GB training corpus.

#### 7.8.5 Two Separate Failure Modes (Do Not Conflate)

| Problem | Symptom | Root cause |
|---|---|---|
| **A. Single-aircraft denoising fails** | 0 frames on `test_capture_36db.bin` (314 raw) | Over-suppression / domain shift; model destroys primary signal structure |
| **B. BSS / collision fails** | Cannot separate overlapping aircraft | No real collision pairs; synthetic augment insufficient for real phase/beating statistics |

Problem **A** must be resolved before Problem **B** is testable on ordinary captures. `test_capture_36db.bin` decodes 314 frames raw with no collision scenario — the immediate blocker is single-aircraft signal destruction, not collision discrimination.

#### 7.8.6 What Has Been Ruled Out

| Hypothesis | Evidence |
|---|---|
| OLA / windowing bug | `--identity` preserves 314/314 frames (§6.6.2) |
| Per-window DC distortion (primary cause) | DC values ~0.001 on this capture; fix did not change frame count |
| Pipeline uint8 corruption | `--passthrough` and `--identity` both preserve decodability |
| "Model never works on real data" | v1 achieves 116 frames on `adsb_capture.bin` |

#### 7.8.7 Recommended Next Steps

1. **Blend diagnostic** — `output = α × model + (1 − α) × raw`; sweep α to find whether model output is directionally useful or fully destructive.
2. **Benchmark v3 on `adsb_capture.bin`** — determine whether v3 exceeds v1's 116-frame in-distribution baseline before further capture-specific debugging.
3. **Capture metadata audit** — compare `test_capture_36db.bin` (1090 MHz, 36 dB gain) against training corpus gain/frequency/site distribution.
4. **Decoder-aware training signal** — weight loss by preamble/data pulse positions or add a downstream CRC-surrogate term so optimisation aligns with frame recovery.
5. **Real collision data (long-term)** — dual-synchronised SDR capture or simulation matching RTL-SDR noise **texture**, not just SNR and frequency offset.

#### 7.6.4 Systemd Unit Consideration

Ensure `RestartSec=5` is set in the unit file to provide a brief pause between the clean exit and the next launch, allowing any lingering OS buffer flushes to complete before a new memmap file is created:

```ini
[Service]
Restart=always
RestartSec=5
```

---

## 8. NEXT PHASES & EVOLUTIONARY ROADMAP

### 8.1 Phase 3 (Current): Expanded Real-World Training Data

**Status: Active data capture.**

The `ml_data_collector.py` service is running continuously on both `rpi_east` and `rpi_west`, uploading to the `adsb-iq-ml` MinIO bucket on `rpi-master` (1 TB SSD at `/mnt/ssd/minio_data`).

**ML collector improvements over the general collector:**
- Burst duration: **30 seconds** (vs. 5 s general) — ~480 frames per burst vs. ~80
- Gain rotation: `[28, 32, 36, 40, 44]` dB — 5× SNR diversity spread
- Gain encoded in filename: `rpi_west_YYYYMMDD_HHMMSS_g040.npy`
- Continuous collection (no inter-interval idle gap)
- Direct upload to separate `adsb-iq-ml` bucket (keeps ML data isolated from general collection)

**406 GB captured across collection sessions (June 2026 — including 35°C heat wave day):**
- ~840 burst files × 2 antennas = ~1,680 × 30-second bursts
- Estimated **~600,000+ supervised pairs** after extraction (80× more than the v1 dataset)
- Heat wave conditions provide additional oscillator drift diversity (temperature-dependent crystal oscillator frequency)
- Collection halted automatically by OOM event on collector Pis — see §7.6

**Label extraction results:** 152,343 supervised pairs extracted from 406 GB across 785 `.npy` files. 2,016 unique ICAO addresses.

**v2 training completed** — best val loss 2.208 (epoch 48/50). Real-world evaluation revealed a regression (8 frames vs. v1's 116) caused by clean-interferer over-suppression. See §6.5 and §7.7 for full analysis.

**v3 training completed** — warm-started from v1 (`best_supervised.pt`), `collision_augment=0.10`, `interferer_snr_db=20` dB. Saved to `checkpoints/best_supervised_v3.pt`. On `test_capture_36db.bin` (out-of-distribution): **0 valid frames** (same as v1/v2). Benchmark on `adsb_capture.bin` (§6.3 in-distribution file) pending. See §6.7 and §7.8 for constraint analysis.

**Pipeline validation completed (July 2026):** DC warmup fix and `--identity` mode confirm inference infrastructure is sound; model weights are the blocker. See §6.6.

### 8.2 Phase 4: Live RTL-SDR → Autoencoder → dump1090 Pipeline

**Implementation: `live_bridge.py`**

Full streaming pipeline:
```
rtl_sdr -f 1090000000 -s 2000000 - \
  | python live_bridge.py --ckpt checkpoints/best_supervised.pt \
  | dump1090 --ifile /dev/stdin --raw
```

Technical components:
- **Overlap-Add (OLA) reconstruction** with Hanning synthesis window (50% overlap default, `hop=120`). OLA blends overlapping model outputs into a continuous stream without window-edge clicks — see §6.6.2 for validation.
- **Batched inference:** `batch_size=32` windows per GPU call (configurable).
- **TorchScript compilation** available via `--torchscript` flag for RPi CPU performance.
- **Global DC warmup:** first ~500 K samples used to estimate hardware `dc_i` / `dc_q` (matches `extract_labels.py` full-burst mean; replaces per-window DC subtraction that distorted pulses). See §6.6.1.
- **Per-window RMS normalisation** auto-detected from checkpoint `training_mode` metadata field (`"supervised_real"` triggers normalisation; synthetic checkpoints bypass it).
- **`--identity` flag:** full pipeline with model replaced by `output = input` — validates OLA/DC/RMS without learned weights (§6.6.2).
- **`--blend ALPHA` flag:** mix model output with raw after OLA: `out = ALPHA × model + (1 − ALPHA) × raw`. Model still runs on 100% of the stream; ALPHA weights the mix only. `0.05` was optimal for v4 on `adsb_capture.bin` (§6.8). Offline equivalent: `blend_sweep.py`.
- **`--passthrough` flag:** uint8→float→uint8 only; skips OLA and windowing (IQ conversion check only).
- **Output:** Standard RTL-SDR `uint8` interleaved IQ binary, directly pipeable to `dump1090`, `dump978`, or any SDR software.

### 8.3 Phase 5: Encoder-Level Architectural Improvements

**Attention mechanism at bottleneck:** Replace the two fixed Conv1d bottleneck layers with a self-attention block operating on the 15 latent time steps. This would allow the model to directly learn which latent positions carry evidence of the interference (e.g., the beating envelope peak positions) and which carry clean preamble information.

**Conditional generation:** Feed the estimated frequency offset \(\hat{f}_{offset}\) (derived from preamble phase) as a conditioning signal to the decoder, allowing it to use an explicit physical prior rather than inferring frequency from the latent alone.

**Multi-frame context:** Extend the input window from 240 to 480 samples (two frames) to provide the model with inter-frame phase continuity context, which would significantly aid in resolving collision cases where the preambles overlap.

### 8.4 Phase 6: Noise2Noise / Self-Supervised Extensions

If dual-antenna temporal-aligned captures become available (same airspace, synchronised timestamps, independent hardware noise):

$$\mathcal{L}_{N2N} = \mathbb{E}\left[\|f(x_{A}) - x_{B}\|^2\right]$$

where \(x_A\) and \(x_B\) are captures from `rpi_east` and `rpi_west` at the same timestamp. The model learns to output the common signal (aircraft transmissions) by being trained to predict one antenna's capture from the other's — without any explicit clean reference. This would eliminate the need for CRC-based label extraction entirely.

---

## 9. SYSTEM FILE INVENTORY

| File | Location | Purpose |
|---|---|---|
| `generator.py` | `adsb_autoencoder/` | Synthetic ADS-B IQ frame synthesis with full impairment modelling |
| `model.py` | `adsb_autoencoder/` | `IQAutoencoder`, `PhaseAwareLoss`, `SyntheticADSBDataset`, training loop |
| `evaluate.py` | `adsb_autoencoder/` | Visual evaluation: constellation plots, magnitude envelope, MSE metrics |
| `live_bridge.py` | `adsb_autoencoder/` | Streaming inference: RTL-SDR binary → model → denoised RTL-SDR binary |
| `extract_labels.py` | `adsb_autoencoder/` | One-time preprocessing: extract supervised pairs from real `.npy` captures |
| `supervised_dataset.py` | `adsb_autoencoder/` | PyTorch Dataset wrapping `*_labels.npz` files |
| `real_dataset.py` | `adsb_autoencoder/` | Self-supervised Dataset for real captures (noise augmentation approach) |
| `train_supervised.py` | `adsb_autoencoder/` | Fine-tuning script: supervised pairs + transfer learning |
| `train_real.py` | `adsb_autoencoder/` | Fine-tuning script: self-supervised noise augmentation approach |
| `compare_decodings.py` | `adsb_autoencoder/` | Evaluation: compare raw vs. denoised CRC-valid frame counts |
| `ml_data_collector.py` | `adsb_autoencoder/` | ML-optimised IQ collector: gain rotation, 30s bursts, MinIO upload |
| `deploy.sh` | `adsb_autoencoder/` | rsync deployment script: Mac → Raspberry Pi |
| `requirements_rpi.txt` | `adsb_autoencoder/` | CPU-only PyTorch wheel index for aarch64 Raspberry Pi |
| `checkpoints/best.pt` | `adsb_autoencoder/` | Synthetic pre-trained checkpoint (base_channels=64, 50 epochs) |
| `checkpoints/best_supervised.pt` | `adsb_autoencoder/` | v1: Real-world fine-tuned (50 epochs, 7,357 pairs) — 116 frames recovered, 7.3% recovery rate |
| `checkpoints/best_supervised_v2.pt` | `adsb_autoencoder/` | v2: Scaled (152K pairs, 30% clean interferer) — regressed to 8 frames; see §7.7 |
| `checkpoints/best_supervised_v3.pt` | `adsb_autoencoder/` | v3: Corrected (10% noisy interferer at 20 dB) — in progress |
| `collector.py` | `adsb_iq_sample_collection/` | General-purpose IQ burst collector (5s, fixed gain) |
| `config_example.py` | `adsb_iq_sample_collection/` | Collector configuration template |

---

*End of Blueprint — Document to be expanded with Phase 3 v3 training results, quantitative improvement metrics, and formal mathematical notation review prior to paper submission.*
