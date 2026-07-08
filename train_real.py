"""
train_real.py — Fine-tune IQAutoencoder on real-world ADS-B IQ captures.

Why fine-tune on real data?
---------------------------
The model trained on synthetic data (train() in model.py) has a domain-shift
problem: the noise statistics, DC bias, and amplitude distribution of real
RTL-SDR captures differ from the synthetic generator.  This script loads the
real .npy bursts collected by adsb_iq_sample_collection and runs a
self-supervised denoising training loop directly on real RF data.

Self-supervised strategy
------------------------
Since no "perfectly clean" ADS-B reference exists for real captures, we use a
Noise2Signal approach (see real_dataset.py):

    noisy_input  = real_signal_window + added_Gaussian_noise   (N SNR-augmented)
    clean_target = real_signal_window  (DC removed)

The model learns to suppress the added noise layer, which generalises to
suppressing the original hardware noise at inference time.

Transfer learning
-----------------
If an existing checkpoint (--warm-start) exists, its weights are loaded before
training begins.  This lets us start from the already-converged synthetic model
and fine-tune rather than training from scratch (typically 5-10x faster to
reach a good real-world loss).

Usage examples
--------------
# Transfer-learn from the synthetic checkpoint on a Mac with MPS:
python train_real.py \\
    --data-dir ~/adsb_iq_data \\
    --warm-start checkpoints/best.pt \\
    --ckpt-out   checkpoints/best_real.pt \\
    --epochs 30

# Train from scratch with a smaller model (faster on RPi CPU):
python train_real.py \\
    --data-dir ~/adsb_iq_data \\
    --base-channels 32 \\
    --ckpt-out checkpoints/best_real.pt \\
    --epochs 20 \\
    --device cpu

# Quick smoke test with a subset of files:
python train_real.py \\
    --data-dir ~/adsb_iq_data \\
    --warm-start checkpoints/best.pt \\
    --max-files 5 \\
    --epochs 5 \\
    --hop 240
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Local modules
from model import (
    AutoencoderConfig,
    IQAutoencoder,
    PhaseAwareLoss,
    _get_device,
    _run_epoch,
)
from real_dataset import RealIQDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune IQAutoencoder on real RTL-SDR IQ captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    data = p.add_argument_group("Data")
    data.add_argument(
        "--data-dir", required=True,
        help="Root directory containing .npy IQ burst files (may include "
             "sub-directories when --recursive is set).",
    )
    data.add_argument(
        "--recursive", action="store_true",
        help="Scan --data-dir recursively for .npy files.",
    )
    data.add_argument(
        "--max-files", type=int, default=None,
        help="Cap the number of .npy files used (for quick experiments).",
    )
    data.add_argument(
        "--hop", type=int, default=120,
        help="Step between consecutive 240-sample windows.  120 = 50%% overlap.",
    )
    data.add_argument(
        "--snr-min", type=float, default=5.0,
        help="Minimum added-noise SNR in dB (lower = harder denoising task).",
    )
    data.add_argument(
        "--snr-max", type=float, default=20.0,
        help="Maximum added-noise SNR in dB.",
    )
    data.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of windows held out for validation (0.0 to skip val).",
    )
    data.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for dataset shuffling and noise augmentation.",
    )

    # ── Model ───────────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model")
    mdl.add_argument(
        "--base-channels", type=int, default=32,
        help="Channel count at the first encoder stage (must match checkpoint "
             "if using --warm-start).  16=light, 32=standard, 64=large.",
    )
    mdl.add_argument(
        "--depth", type=int, default=4,
        help="Number of encoder/decoder stages (must match --warm-start).",
    )
    mdl.add_argument(
        "--warm-start", type=str, default=None,
        metavar="CKPT",
        help="Path to an existing checkpoint (.pt) to use as initial weights "
             "(transfer learning).  Highly recommended: use checkpoints/best.pt "
             "from the synthetic-data run.",
    )
    mdl.add_argument(
        "--freeze-encoder", action="store_true",
        help="Freeze encoder weights for the first --freeze-epochs epochs, "
             "then unfreeze for full fine-tuning.  Helps when the synthetic "
             "encoder features are already good.",
    )
    mdl.add_argument(
        "--freeze-epochs", type=int, default=5,
        help="Number of initial epochs to train only the decoder (only used "
             "with --freeze-encoder).",
    )

    # ── Training ────────────────────────────────────────────────────────────
    trn = p.add_argument_group("Training")
    trn.add_argument("--epochs", type=int, default=30)
    trn.add_argument("--batch-size", type=int, default=128)
    trn.add_argument("--lr", type=float, default=3e-4,
                     help="Peak learning rate for Adam.")
    trn.add_argument("--weight-decay", type=float, default=1e-4)
    trn.add_argument("--grad-clip", type=float, default=1.0)
    trn.add_argument(
        "--loss-weights", type=float, nargs=3, default=[1.0, 0.5, 0.3],
        metavar=("W_IQ", "W_MAG", "W_PHASE"),
        help="Weights for the PhaseAwareLoss components.",
    )
    trn.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes.  0 = main process only (safe on "
             "RPi; set 2-4 on Mac for faster data loading).",
    )

    # ── Output ──────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument(
        "--ckpt-out", type=str, default="checkpoints/best_real.pt",
        help="Where to save the best checkpoint.",
    )
    out.add_argument(
        "--history-out", type=str, default=None,
        help="Optional path to save training history JSON (e.g. history_real.json).",
    )
    out.add_argument(
        "--device", type=str, default=None,
        help="Force device: 'cpu', 'cuda', 'mps'.  Auto-detect if omitted.",
    )

    return p


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_datasets(args: argparse.Namespace) -> Tuple[RealIQDataset, Optional[RealIQDataset]]:
    """Load RealIQDataset and optionally split off a validation subset."""
    full_ds = RealIQDataset(
        data_dir=args.data_dir,
        hop=args.hop,
        noise_snr_db_range=(args.snr_min, args.snr_max),
        max_files=args.max_files,
        recursive=args.recursive,
        seed=args.seed,
        dc_remove=True,
    )

    if args.val_fraction <= 0.0 or args.val_fraction >= 1.0:
        return full_ds, None

    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    print(f"Split: {n_train:,} train  |  {n_val:,} val")
    return train_ds, val_ds  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Encoder freeze helpers
# ---------------------------------------------------------------------------

def _set_encoder_grad(model: IQAutoencoder, requires_grad: bool) -> None:
    """Enable or disable gradients for encoder (and bottleneck) weights."""
    for name, param in model.named_parameters():
        if name.startswith("encoder") or name.startswith("bottleneck"):
            param.requires_grad_(requires_grad)


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Warm-start loader
# ---------------------------------------------------------------------------

def _warm_start(
    model: IQAutoencoder,
    ckpt_path: str,
    device: torch.device,
    verbose: bool = True,
) -> None:
    """
    Load weights from an existing checkpoint into ``model``.

    Only tensors whose name AND shape both match are transferred.  Size
    mismatches (e.g. different base_channels) result in those layers being
    skipped with a warning rather than a hard crash, enabling partial
    transfer even when architectures differ slightly.
    """
    path = Path(ckpt_path)
    if not path.exists():
        print(f"[warm-start] WARNING: checkpoint not found at {path} — training from scratch.")
        return

    ckpt = torch.load(path, map_location=device, weights_only=False)
    src_state = ckpt.get("model_state", ckpt)  # handle bare state_dict saves
    dst_state = model.state_dict()

    # Keep only keys that exist in the model AND have matching shapes.
    compatible = {
        k: v
        for k, v in src_state.items()
        if k in dst_state and dst_state[k].shape == v.shape
    }
    skipped_shape = [
        k for k, v in src_state.items()
        if k in dst_state and dst_state[k].shape != v.shape
    ]
    missing = [k for k in dst_state if k not in src_state]

    dst_state.update(compatible)
    model.load_state_dict(dst_state, strict=True)

    if verbose:
        ckpt_cfg = ckpt.get("config", {})
        print(f"[warm-start] Loaded       : {path}")
        print(f"             Ckpt config  : base_channels={ckpt_cfg.get('base_channels', '?')}  "
              f"depth={ckpt_cfg.get('depth', '?')}")
        print(f"             Transferred  : {len(compatible)} / {len(dst_state)} tensors")
        if skipped_shape:
            print(f"             Shape mismatch (skipped): {len(skipped_shape)} tensors — "
                  "model will train those layers from random init")
        if missing:
            print(f"             Not in ckpt  : {len(missing)} tensors (random init)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _format_loss(total: float, components: Dict[str, float], elapsed: float) -> str:
    return (
        f"total={total:.5f}  "
        f"iq={components['l_iq']:.5f}  "
        f"mag={components['l_mag']:.5f}  "
        f"phase={components['l_phase']:.5f}  "
        f"({elapsed:.1f}s)"
    )


def run_training(args: argparse.Namespace) -> Dict[str, List[float]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = _get_device(args.device)
    print(f"Device : {dev}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds, val_ds = _build_datasets(args)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs) if val_ds else None

    # ── Model ─────────────────────────────────────────────────────────────
    config = AutoencoderConfig(
        seq_len=240,
        in_channels=2,
        base_channels=args.base_channels,
        depth=args.depth,
    )
    model = IQAutoencoder(config=config).to(dev)

    if args.warm_start:
        _warm_start(model, args.warm_start, dev)

    loss_fn = PhaseAwareLoss(
        w_iq=args.loss_weights[0],
        w_mag=args.loss_weights[1],
        w_phase=args.loss_weights[2],
    )

    # ── Encoder freeze setup ───────────────────────────────────────────────
    encoder_frozen = False
    if args.freeze_encoder:
        _set_encoder_grad(model, requires_grad=False)
        encoder_frozen = True
        print(
            f"[freeze] Encoder frozen for first {args.freeze_epochs} epoch(s).  "
            f"Trainable params: {_count_trainable(model):,}"
        )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100.0
    )

    # ── Print model summary ───────────────────────────────────────────────
    print(f"Model  : IQAutoencoder  |  {model.count_parameters():,} params total  "
          f"|  {_count_trainable(model):,} trainable")
    lat_ch, lat_t = model.latent_shape()
    print(f"         channels {config.channel_schedule}  →  latent ({lat_ch}, {lat_t})")
    print(f"Loss   : PhaseAwareLoss  "
          f"w_iq={args.loss_weights[0]}  "
          f"w_mag={args.loss_weights[1]}  "
          f"w_phase={args.loss_weights[2]}")
    print(f"Optim  : Adam  lr={args.lr}  weight_decay={args.weight_decay}")
    print(f"{'─' * 70}")

    # ── Checkpoint setup ──────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_l_iq": [], "train_l_mag": [], "train_l_phase": [],
        "val_l_iq":   [], "val_l_mag":   [], "val_l_phase":   [],
        "lr": [],
    }

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder after freeze_epochs
        if encoder_frozen and epoch > args.freeze_epochs:
            _set_encoder_grad(model, requires_grad=True)
            encoder_frozen = False
            # Rebuild optimizer to include newly unfrozen parameters
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1, eta_min=args.lr / 100.0
            )
            print(f"[freeze] Encoder unfrozen at epoch {epoch}.  "
                  f"Trainable params: {_count_trainable(model):,}")

        t0 = time.time()

        train_loss, train_comp = _run_epoch(
            model, train_loader, loss_fn, optimizer, dev, grad_clip=args.grad_clip
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["lr"].append(current_lr)
        for k in ("l_iq", "l_mag", "l_phase"):
            history[f"train_{k}"].append(train_comp[k])

        val_tag = ""
        if val_loader is not None:
            val_loss, val_comp = _run_epoch(
                model, val_loader, loss_fn, optimizer=None, device=dev
            )
            history["val_loss"].append(val_loss)
            for k in ("l_iq", "l_mag", "l_phase"):
                history[f"val_{k}"].append(val_comp[k])

            monitor_loss = val_loss
            val_summary = f"  |  val {_format_loss(val_loss, val_comp, 0.0)}"
        else:
            monitor_loss = train_loss
            val_summary = ""

        # ── Save best checkpoint ─────────────────────────────────────────
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            torch.save(
                {
                    "epoch": epoch,
                    "config": asdict(model.config),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": monitor_loss,
                    "training_mode": "real_world",
                },
                ckpt_path,
            )
            val_tag = "  ✓ best"

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train {_format_loss(train_loss, train_comp, elapsed)}"
            f"{val_summary}{val_tag}"
        )

    print(f"{'─' * 70}")
    print(f"Training complete.  Best loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved → {ckpt_path}")

    # ── Save history ──────────────────────────────────────────────────────
    if args.history_out:
        hist_path = Path(args.history_out)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History  saved → {hist_path}")

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not Path(args.data_dir).exists():
        print(
            f"ERROR: --data-dir '{args.data_dir}' does not exist.\n"
            "Copy .npy burst files from the Pi first, e.g.:\n"
            "  rsync -av pi@raspberrypi.local:~/workspace/adsb_iq_collector/data/ "
            "~/adsb_iq_data/",
            file=sys.stderr,
        )
        sys.exit(1)

    run_training(args)


if __name__ == "__main__":
    main()
