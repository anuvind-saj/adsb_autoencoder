"""
model.py — Phase-Aware 1D-CNN Autoencoder for ADS-B IQ Signal Processing
=========================================================================
Implements a U-Net style 1D convolutional autoencoder that operates directly
on raw complex baseband IQ data.  The network learns the geometric structure
of valid ADS-B signals on the complex unit circle, enabling:

  1. Phase-aware denoising — separate thermal/quantisation noise from the
                             continuous phase arc of a valid transmission.
  2. Collision resolution  — reconstruct the primary aircraft signal from a
                             two-signal composite by exploiting the distinct
                             frequency offsets (different spiral rates on the
                             complex plane — the BSS discriminant).

Architecture: U-Net 1D-CNN Autoencoder
-----------------------------------------------------------------------
Input    : (B, 2, 240)  — batch × {I, Q} × 240 samples  (120 µs @ 2 MHz)

Encoder  : 4 stride-2 stages, channel depth doubles at each stage.
           Each stage = Conv1d(stride=2) → BN → LeakyReLU
                      + Conv1d(stride=1) → BN → LeakyReLU   (feature refine)
           Resolutions: 240 → 120 → 60 → 30 → 15
           Channels   :   2 →  32 → 64 → 128 → 256

Bottleneck: 2 × Conv1d(256, 256, kernel=3) at (B, 256, 15) — deepens
            feature representation at the most compressed resolution.

Decoder  : 4 stride-2 ConvTranspose1d stages with U-Net skip connections
           from matching encoder resolutions.  Each stage:
             ConvTranspose1d(stride=2) → BN → LeakyReLU
             cat(skip)
             Conv1d(in+skip → out, stride=1) → BN → LeakyReLU

Output   : Conv1d(base_channels, 2, kernel=1) → raw (I, Q) reconstruction.
           No final activation — IQ values are unbounded real numbers.

Loss Function: PhaseAwareLoss
-----------------------------------------------------------------------
  L_total = w_iq    × MSE(IQ_pred, IQ_clean)           — raw channel fidelity
          + w_mag   × MSE(|IQ_pred|, |IQ_clean|)        — magnitude envelope
          + w_phase × E[ (1 − cos Δφ) × w_signal ]      — circular phase error
                                                           masked by signal power

  where Δφ = atan2(Q_pred, I_pred) − atan2(Q_clean, I_clean)
  and   w_signal = |IQ_clean| / (mean|IQ_clean| + ε)  ← zero-weight during silence

  The (1 − cos Δφ) formulation avoids wrap-around issues and is equivalent to
  the squared angular chord distance on the unit circle.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AutoencoderConfig:
    """
    All architecture hyper-parameters in one place.

    Attributes
    ----------
    seq_len          : IQ sequence length.  Must be divisible by 2 ** depth.
                       Default 240 = 16 × 15.
    in_channels      : Number of input channels (2 = I and Q).
    base_channels    : Channel count at the first encoder stage.  Each
                       subsequent stage doubles this up to depth stages.
                       Controls model capacity: 16 ≈ light, 32 ≈ standard,
                       64 ≈ large.
    depth            : Number of encoder / decoder stages (= number of 2×
                       downsampling steps).  Default 4 gives a compression
                       ratio of 2^4 = 16 along the time axis.
    bottleneck_layers: Number of Conv1d layers in the bottleneck block at
                       the most compressed resolution.
    leaky_slope      : Negative slope for LeakyReLU activations.
    dropout          : Dropout probability applied after the bottleneck.
                       0.0 disables dropout.
    """

    seq_len: int = 240
    in_channels: int = 2
    base_channels: int = 32
    depth: int = 4
    bottleneck_layers: int = 2
    leaky_slope: float = 0.2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.seq_len % (2 ** self.depth) != 0:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be divisible by "
                f"2**depth = {2 ** self.depth}."
            )

    @property
    def channel_schedule(self) -> List[int]:
        """Channel widths at each stage: [in_channels, base, 2*base, ...]."""
        return [self.in_channels] + [
            self.base_channels * (2 ** i) for i in range(self.depth)
        ]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """
    Single encoder stage: stride-2 downsampling + stride-1 feature refine.

    Spatial resolution halves; channel count increases.

    Input  : (B, in_ch,  L)
    Output : (B, out_ch, L // 2)
    """

    def __init__(self, in_ch: int, out_ch: int, leaky_slope: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # Stride-2 conv: halves the time axis
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(leaky_slope, inplace=True),
            # Stride-1 conv: deepens feature representation at the new resolution
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """
    Single decoder stage: stride-2 upsampling + skip connection fusion.

    The U-Net skip connection concatenates the encoder feature map at the
    matching resolution, then a fusion conv recombines them.  This lets the
    decoder recover fine-grained temporal detail that the bottleneck would
    otherwise lose.

    Input  : x       (B, in_ch,       L)
             skip    (B, skip_ch,     L * 2)   — from matching encoder stage
    Output :         (B, out_ch,      L * 2)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        skip_ch: int,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()

        # ConvTranspose1d doubles the time axis.
        # output_padding=1 is required so that the output length matches exactly:
        #   L_out = (L_in − 1) × 2 − 2 × 1 + 3 + 1 = 2 × L_in
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(
                in_ch, out_ch,
                kernel_size=3, stride=2, padding=1, output_padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # Fusion conv after skip concatenation
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)   # concatenate along channel axis
        return self.fuse(x)


class _BottleneckBlock(nn.Module):
    """
    Multi-layer 1D conv block at the most compressed resolution (B, 256, 15).

    Operates at fixed spatial resolution (no stride), allowing the network to
    learn longer-range interactions across the 15-sample latent time axis.
    This is where the model integrates evidence from across the entire frame
    to disentangle the two collision sources.
    """

    def __init__(
        self,
        channels: int,
        n_layers: int = 2,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(leaky_slope, inplace=True),
            ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class IQAutoencoder(nn.Module):
    """
    U-Net style 1D-CNN Autoencoder for complex IQ signal denoising.

    Input  : (B, 2, seq_len)
    Output : (B, 2, seq_len)  — reconstructed clean IQ frame

    The encoder progressively compresses the time axis while expanding the
    channel axis.  Skip connections pass encoder feature maps to the decoder
    at each matching resolution.  The bottleneck integrates global frame
    context for collision resolution.

    Parameters
    ----------
    config : AutoencoderConfig or None.  None uses default parameters.

    Example
    -------
    >>> model = IQAutoencoder()
    >>> x = torch.randn(8, 2, 240)   # batch of 8 noisy IQ frames
    >>> y = model(x)                  # reconstructed clean frames
    >>> y.shape
    torch.Size([8, 2, 240])
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = AutoencoderConfig()
        self.config = config

        ch = config.channel_schedule   # e.g. [2, 32, 64, 128, 256]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.ModuleList([
            _EncoderBlock(ch[i], ch[i + 1], config.leaky_slope)
            for i in range(config.depth)
        ])

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = _BottleneckBlock(
            channels=ch[-1],
            n_layers=config.bottleneck_layers,
            leaky_slope=config.leaky_slope,
            dropout=config.dropout,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # Each decoder stage upsamples and fuses the skip from the matching
        # encoder stage (in reverse order).
        self.decoder = nn.ModuleList([
            _DecoderBlock(
                in_ch=ch[config.depth - i],        # e.g. 256, 128, 64, 32
                out_ch=ch[config.depth - i - 1],   # e.g. 128,  64, 32, 32
                skip_ch=ch[config.depth - i - 1],  # matching encoder output
                leaky_slope=config.leaky_slope,
            )
            for i in range(config.depth - 1)
        ])

        # Final upsample from the smallest decoder output up to base_channels
        # resolution, WITHOUT a skip (the input has only 2 channels, not
        # base_channels, so there is no meaningful skip at this level).
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(
                ch[1], ch[1],
                kernel_size=3, stride=2, padding=1, output_padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ch[1]),
            nn.LeakyReLU(config.leaky_slope, inplace=True),
        )

        # ── Output head ──────────────────────────────────────────────────────
        # 1×1 conv maps base_channels → 2 (I and Q).  No activation — the
        # network should output unbounded real-valued IQ coordinates.
        self.output_head = nn.Conv1d(ch[1], config.in_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for conv layers; ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, a=self.config.leaky_slope,
                                        mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, 2, seq_len) float32 — noisy / collided IQ input.

        Returns
        -------
        (B, 2, seq_len) float32 — reconstructed clean IQ estimate.
        """
        # ── Encode ───────────────────────────────────────────────────────────
        skips: List[torch.Tensor] = []
        for enc_block in self.encoder:
            x = enc_block(x)
            skips.append(x)   # save feature map before next downsampling

        # ── Bottleneck ───────────────────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decode with skip connections (all but the last encoder skip) ─────
        # skips[-1] is the deepest encoder output (same resolution as bottleneck
        # output), so we start fusing from skips[-2] downward.
        for i, dec_block in enumerate(self.decoder):
            skip = skips[-(i + 2)]   # skip at next coarser resolution
            x = dec_block(x, skip)

        # ── Final upsample to full resolution ────────────────────────────────
        x = self.final_upsample(x)

        # ── Output projection ─────────────────────────────────────────────────
        return self.output_head(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def latent_shape(self) -> Tuple[int, int]:
        """Return (channels, time_steps) of the bottleneck representation."""
        ch = self.config.channel_schedule
        return ch[-1], self.config.seq_len // (2 ** self.config.depth)


# ---------------------------------------------------------------------------
# Phase-Aware Loss Function
# ---------------------------------------------------------------------------

class PhaseAwareLoss(nn.Module):
    """
    Composite loss that penalises reconstruction error in three complementary
    geometric spaces of the complex plane:

      L_iq    : Mean Squared Error on raw I and Q channels.
                This is the workhorse loss — it drives the model to reproduce
                the exact IQ waveform including all pulse positions.

      L_mag   : MSE between the predicted and target magnitude envelopes
                √(I² + Q²).  This forces the model to correctly reconstruct
                which samples are "on" (near unit circle) and "off" (near
                origin), regardless of the phase.  Critical for preamble
                structure recovery.

      L_phase : Circular phase error, computed via:

                  Δφ  = atan2(Q_pred, I_pred) − atan2(Q_target, I_target)
                  err = 1 − cos(Δφ)    ∈ [0, 2]

                The (1 − cos) formulation is wrap-around safe and equals the
                squared chord distance on the unit circle.  Only penalises
                samples where the target magnitude is significant — silence
                periods have undefined phase and are masked out via a
                magnitude-proportional weight.

    Total loss:
        L = w_iq × L_iq  +  w_mag × L_mag  +  w_phase × L_phase

    Parameters
    ----------
    w_iq    : Weight on the channel MSE term.   Default 1.0.
    w_mag   : Weight on the magnitude MSE term. Default 0.5.
    w_phase : Weight on the phase error term.   Default 0.3.
    eps     : Small constant for numerical stability in sqrt and division.
    """

    def __init__(
        self,
        w_iq: float = 1.0,
        w_mag: float = 0.5,
        w_phase: float = 0.3,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.w_iq = w_iq
        self.w_mag = w_mag
        self.w_phase = w_phase
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the composite loss.

        Parameters
        ----------
        pred   : (B, 2, N) — network output.  Channel 0 = I, Channel 1 = Q.
        target : (B, 2, N) — clean ground-truth IQ.

        Returns
        -------
        total_loss : scalar tensor (autograd-connected).
        components : dict with keys 'l_iq', 'l_mag', 'l_phase' for logging.
        """
        I_pred, Q_pred = pred[:, 0, :], pred[:, 1, :]
        I_tgt,  Q_tgt  = target[:, 0, :], target[:, 1, :]

        # ── 1. Channel MSE ────────────────────────────────────────────────
        l_iq = F.mse_loss(pred, target)

        # ── 2. Magnitude MSE ─────────────────────────────────────────────
        # eps inside the sqrt prevents gradient explosion at exactly (0, 0).
        mag_pred = torch.sqrt(I_pred ** 2 + Q_pred ** 2 + self.eps)
        mag_tgt  = torch.sqrt(I_tgt  ** 2 + Q_tgt  ** 2 + self.eps)
        l_mag = F.mse_loss(mag_pred, mag_tgt)

        # ── 3. Circular phase error ────────────────────────────────────────
        # atan2 gives the instantaneous carrier phase at each sample.
        phi_pred = torch.atan2(Q_pred, I_pred)   # (B, N), range (−π, π]
        phi_tgt  = torch.atan2(Q_tgt,  I_tgt)

        # (1 − cos Δφ) is wrap-around safe: cos(π) = −1 → error = 2 (max),
        # cos(0) = 1 → error = 0 (perfect).
        phase_err = 1.0 - torch.cos(phi_pred - phi_tgt)   # (B, N), ∈ [0, 2]

        # Magnitude-proportional weight: ~1 for on-state samples, ~0 for
        # silence.  Dividing by the batch mean normalises the scale so that
        # the phase weight is dataset-independent.
        signal_weight = mag_tgt / (mag_tgt.mean() + self.eps)
        l_phase = (phase_err * signal_weight).mean()

        # ── Combine ───────────────────────────────────────────────────────
        total = (
            self.w_iq    * l_iq
          + self.w_mag   * l_mag
          + self.w_phase * l_phase
        )

        components = {
            "l_iq":    l_iq.detach(),
            "l_mag":   l_mag.detach(),
            "l_phase": l_phase.detach(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticADSBDataset(Dataset):
    """
    In-memory PyTorch Dataset of synthetic ADS-B IQ frames.

    Generates all samples once at construction using ADSBSignalGenerator and
    stores them as float32 tensors.  Each __getitem__ call returns a
    (noisy_iq, clean_iq) pair.

    Parameters
    ----------
    n_samples         : Number of (noisy, clean) pairs to generate.
    snr_db_range      : (min, max) SNR range in dB for AWGN.
    f_offset_range    : (min, max) carrier frequency offset range in Hz.
    include_collisions: Mix collision scenarios into the dataset.
    collision_fraction: Fraction of samples that are two-signal collisions.
    phase_noise_rad   : Per-sample carrier phase jitter std-dev (rad).
    seed              : RNG seed for reproducible datasets.
    """

    def __init__(
        self,
        n_samples: int = 8192,
        snr_db_range: Tuple[float, float] = (8.0, 25.0),
        f_offset_range: Tuple[float, float] = (-50_000.0, 50_000.0),
        include_collisions: bool = True,
        collision_fraction: float = 0.25,
        phase_noise_rad: float = 0.025,
        seed: int = 42,
    ) -> None:
        super().__init__()

        # Import generator here so model.py can still be imported even if
        # generator.py is not in sys.path (e.g. during unit testing of model
        # classes only).
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from generator import ADSBSignalGenerator
        except ImportError as exc:
            raise ImportError(
                "ADSBSignalGenerator not found.  Ensure generator.py is in "
                "the same directory as model.py."
            ) from exc

        gen = ADSBSignalGenerator(rng_seed=seed)
        noisy_np, clean_np = gen.generate_batch(
            batch_size=n_samples,
            snr_db_range=snr_db_range,
            f_offset_range=f_offset_range,
            include_collisions=include_collisions,
            collision_fraction=collision_fraction,
            phase_noise_rad=phase_noise_rad,
        )
        # Store as float32 tensors on CPU — DataLoader workers will handle
        # device transfer via the collate_fn / pin_memory mechanism.
        self.noisy = torch.from_numpy(noisy_np)   # (N, 2, 240)
        self.clean = torch.from_numpy(clean_np)   # (N, 2, 240)

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _get_device(device: Optional[str]) -> torch.device:
    """Select the best available compute device with cross-platform fallback.

    Priority: explicit arg → CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU.

    Platform behaviour
    ------------------
    macOS + Apple Silicon : auto-selects MPS for GPU-accelerated training.
    Linux x86_64 + NVIDIA : auto-selects CUDA.
    Linux ARM64 (RPi 4/5) : MPS and CUDA are both unavailable; falls back to
                            CPU automatically.  Call
                            ``torch.set_num_threads(N)`` before training to
                            exploit all N cores of the Pi's CPU.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _format_loss(
    total: float,
    components: Dict[str, float],
    elapsed: float,
) -> str:
    return (
        f"total={total:.5f}  "
        f"iq={components['l_iq']:.5f}  "
        f"mag={components['l_mag']:.5f}  "
        f"phase={components['l_phase']:.5f}  "
        f"({elapsed:.1f}s)"
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: PhaseAwareLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Run one full pass over `loader` (train if optimizer given, else eval)."""
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    sum_components: Dict[str, float] = {"l_iq": 0.0, "l_mag": 0.0, "l_phase": 0.0}
    n_batches = 0

    with ctx:
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            pred = model(noisy)
            loss, comp = loss_fn(pred, clean)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            for k in sum_components:
                sum_components[k] += comp[k].item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_comp = {k: v / max(n_batches, 1) for k, v in sum_components.items()}
    return avg_loss, avg_comp


def train(
    n_epochs: int = 50,
    batch_size: int = 128,
    train_samples: int = 8192,
    val_samples: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    snr_db_range: Tuple[float, float] = (8.0, 25.0),
    f_offset_range: Tuple[float, float] = (-50_000.0, 50_000.0),
    include_collisions: bool = True,
    collision_fraction: float = 0.25,
    phase_noise_rad: float = 0.025,
    loss_weights: Tuple[float, float, float] = (1.0, 0.5, 0.3),
    model_config: Optional[AutoencoderConfig] = None,
    checkpoint_dir: str = "checkpoints",
    device: Optional[str] = None,
    seed: int = 42,
    num_workers: int = 0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Full training pipeline for the IQAutoencoder.

    Instantiates the generator, builds synthetic train/val datasets, and
    runs the Adam optimiser with a CosineAnnealingLR schedule.  The best
    validation-loss checkpoint is saved to `checkpoint_dir/best.pt`.

    Parameters
    ----------
    n_epochs          : Number of training epochs.
    batch_size        : Mini-batch size for train and val loaders.
    train_samples     : Number of synthetic training samples generated once
                        at the start of training.
    val_samples       : Number of validation samples (different RNG seed).
    lr                : Peak learning rate for Adam.
    weight_decay      : L2 regularisation coefficient for Adam.
    snr_db_range      : (min, max) SNR range in dB.
    f_offset_range    : (min, max) carrier frequency offset (Hz).
    include_collisions: Include two-signal collision scenarios.
    collision_fraction: Fraction of training samples that are collisions.
    phase_noise_rad   : Per-sample phase jitter for the generator.
    loss_weights      : (w_iq, w_mag, w_phase) for PhaseAwareLoss.
    model_config      : AutoencoderConfig (None = defaults).
    checkpoint_dir    : Directory for saving checkpoints.
    device            : 'cpu', 'cuda', 'mps', or None (auto-detect).
    seed              : RNG seed for reproducible dataset generation.
    num_workers       : DataLoader worker count (0 = main process only,
                        safe for in-memory datasets).
    verbose           : Print epoch summaries to stdout.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss' (lists of floats, one
              per epoch) and sub-component lists 'train_l_iq', 'train_l_mag',
              'train_l_phase', 'val_l_iq', 'val_l_mag', 'val_l_phase'.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = _get_device(device)
    if verbose:
        print(f"Device : {dev}")

    # ── Datasets and loaders ─────────────────────────────────────────────────
    if verbose:
        print(f"Building datasets  (train={train_samples}, val={val_samples}) …")

    dataset_kwargs = dict(
        snr_db_range=snr_db_range,
        f_offset_range=f_offset_range,
        include_collisions=include_collisions,
        collision_fraction=collision_fraction,
        phase_noise_rad=phase_noise_rad,
    )
    train_ds = SyntheticADSBDataset(n_samples=train_samples, seed=seed,       **dataset_kwargs)
    val_ds   = SyntheticADSBDataset(n_samples=val_samples,   seed=seed + 999, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(dev.type == "cuda"),
    )

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    model = IQAutoencoder(config=model_config).to(dev)
    loss_fn = PhaseAwareLoss(
        w_iq=loss_weights[0], w_mag=loss_weights[1], w_phase=loss_weights[2]
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 100.0
    )

    if verbose:
        ch = model.config.channel_schedule
        lat_ch, lat_t = model.latent_shape()
        print(f"Model  : IQAutoencoder  |  {model.count_parameters():,} params")
        print(f"         channels {ch}  →  latent ({lat_ch}, {lat_t})")
        print(f"Loss   : PhaseAwareLoss  w_iq={loss_weights[0]}  "
              f"w_mag={loss_weights[1]}  w_phase={loss_weights[2]}")
        print(f"Optim  : Adam  lr={lr}  weight_decay={weight_decay}")
        print(f"{'─' * 70}")

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"
    best_val_loss = float("inf")

    # ── History ──────────────────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_l_iq": [], "train_l_mag": [], "train_l_phase": [],
        "val_l_iq":   [], "val_l_mag":   [], "val_l_phase":   [],
        "lr": [],
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_comp = _run_epoch(
            model, train_loader, loss_fn, optimizer, dev
        )
        val_loss, val_comp = _run_epoch(
            model, val_loader, loss_fn, optimizer=None, device=dev
        )

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── Record history ────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for k in ("l_iq", "l_mag", "l_phase"):
            history[f"train_{k}"].append(train_comp[k])
            history[f"val_{k}"].append(val_comp[k])

        # ── Checkpoint ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "config": asdict(model.config),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt,
            )
            tag = " ✓ best"
        else:
            tag = ""

        # ── Print ──────────────────────────────────────────────────────
        if verbose:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3}/{n_epochs}  "
                f"train {_format_loss(train_loss, train_comp, elapsed)}  |  "
                f"val {_format_loss(val_loss, val_comp, 0.0)}{tag}"
            )

    if verbose:
        print(f"{'─' * 70}")
        print(f"Training complete.  Best val loss: {best_val_loss:.6f}")
        print(f"Checkpoint saved → {best_ckpt}")

    return history


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_model(
    model: IQAutoencoder,
    path: str | Path,
    extra: Optional[dict] = None,
) -> None:
    """
    Save a self-contained model checkpoint.

    The saved dict includes the AutoencoderConfig so the model can be
    reconstructed without access to the original config object.
    """
    payload = {
        "config": asdict(model.config),
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path))


def load_model(path: str | Path, device: Optional[str] = None) -> IQAutoencoder:
    """
    Load a model from a checkpoint saved by `save_model` or `train`.

    Parameters
    ----------
    path   : Path to the .pt checkpoint file.
    device : Target device string ('cpu', 'cuda', 'mps', or None = auto).

    Returns
    -------
    model : IQAutoencoder in eval() mode on the requested device.
    """
    dev = _get_device(device)
    payload = torch.load(path, map_location=dev)
    config = AutoencoderConfig(**payload["config"])
    model = IQAutoencoder(config=config).to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Quick demo / smoke-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Smoke-test all components without a full training run:
      1. Model construction and parameter count.
      2. Forward pass shape verification.
      3. Loss function components.
      4. Mini training run (5 epochs, small dataset) to confirm convergence.
    """
    sep = "=" * 70

    print(sep)
    print("1 — Model construction")
    print(sep)
    for base_ch in (16, 32, 64):
        cfg = AutoencoderConfig(base_channels=base_ch)
        m = IQAutoencoder(cfg)
        lat_ch, lat_t = m.latent_shape()
        print(
            f"  base_channels={base_ch:>2}  "
            f"channels={cfg.channel_schedule}  "
            f"latent=({lat_ch}, {lat_t})  "
            f"params={m.count_parameters():>9,}"
        )

    print(f"\n{sep}")
    print("2 — Forward pass shapes")
    print(sep)
    model = IQAutoencoder()
    model.eval()
    with torch.no_grad():
        x = torch.randn(8, 2, 240)
        y = model(x)
    print(f"  Input  : {list(x.shape)}")
    print(f"  Output : {list(y.shape)}")
    assert y.shape == x.shape, "Shape mismatch!"
    print(f"  [OK] Output shape matches input.")

    print(f"\n{sep}")
    print("3 — Loss function")
    print(sep)
    loss_fn = PhaseAwareLoss(w_iq=1.0, w_mag=0.5, w_phase=0.3)
    pred   = torch.randn(4, 2, 240)
    target = torch.randn(4, 2, 240)
    total, comps = loss_fn(pred, target)
    print(f"  Random pred vs target:  total={total.item():.4f}  "
          f"iq={comps['l_iq'].item():.4f}  "
          f"mag={comps['l_mag'].item():.4f}  "
          f"phase={comps['l_phase'].item():.4f}")
    # Perfect prediction should give near-zero loss
    total_perf, _ = loss_fn(target, target)
    assert total_perf.item() < 1e-5, f"Perfect prediction loss too high: {total_perf.item()}"
    print(f"  Perfect reconstruction: total={total_perf.item():.2e}  [OK]")

    print(f"\n{sep}")
    print("4 — Mini training run (5 epochs, 512 samples)")
    print(sep)
    history = train(
        n_epochs=5,
        batch_size=64,
        train_samples=512,
        val_samples=128,
        lr=1e-3,
        model_config=AutoencoderConfig(base_channels=16),  # small model for speed
        checkpoint_dir="checkpoints_demo",
        seed=0,
        verbose=True,
    )
    # Verify loss went down
    assert history["train_loss"][-1] < history["train_loss"][0], \
        "Training loss did not decrease!"
    print(f"\n  [OK] Training loss reduced: "
          f"{history['train_loss'][0]:.5f} → {history['train_loss'][-1]:.5f}")

    print(f"\n{sep}")
    print("5 — Checkpoint save/load round-trip")
    print(sep)
    model = IQAutoencoder(AutoencoderConfig(base_channels=16))
    ckpt_path = Path("checkpoints_demo") / "roundtrip_test.pt"
    save_model(model, ckpt_path)
    loaded = load_model(ckpt_path)   # load_model() puts the model in eval()
    model.eval()                     # match mode for BatchNorm consistency
    with torch.no_grad():
        x_t = torch.randn(2, 2, 240)
        out_orig   = model(x_t)
        out_loaded = loaded(x_t)
    diff = (out_orig - out_loaded).abs().max().item()
    assert diff < 1e-5, f"Round-trip mismatch: {diff}"
    print(f"  Saved  → {ckpt_path}")
    print(f"  Loaded → max output diff = {diff:.2e}  [OK]")


if __name__ == "__main__":
    _demo()
