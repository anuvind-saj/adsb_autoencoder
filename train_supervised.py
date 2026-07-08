"""
train_supervised.py — Fine-tune IQAutoencoder on real-world supervised pairs.

Training data
-------------
Uses pairs produced by extract_labels.py:
  noisy_input  = real RTL-SDR captured IQ window (DC removed)
  clean_target = re-synthesized perfect IQ from decoded 112-bit payload,
                 aligned in amplitude, phase, and frequency to the real window.

This is the highest-quality training signal available: the model is explicitly
shown what a perfect ADS-B frame looks like for every noisy input it receives.

Workflow
--------
  Step 1 — Extract labels (one time):
    python extract_labels.py \\
        --data-dir  ~/adsb_iq_data \\
        --labels-dir ~/adsb_iq_data/labels \\
        --recursive

  Step 2 — Fine-tune (Mac with MPS, recommended):
    python train_supervised.py \\
        --labels-dir ~/adsb_iq_data/labels \\
        --warm-start checkpoints/best.pt \\
        --ckpt-out   checkpoints/best_supervised.pt \\
        --epochs 50

  Step 3 — Deploy the new checkpoint:
    Update live_bridge.py --ckpt argument to checkpoints/best_supervised.pt
    and re-run the Python decoder comparison.

Transfer learning
-----------------
Always pass --warm-start checkpoints/best.pt to initialise from the synthetic
pre-training.  The synthetic model has already learned:
  - preamble shape recognition
  - PPM bit timing
  - phase continuity on the complex plane

Fine-tuning on real pairs adapts the noise-suppression layers to the actual
RTL-SDR noise statistics in a few epochs, rather than training from scratch.
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

from model import (
    AutoencoderConfig,
    IQAutoencoder,
    PhaseAwareLoss,
    _get_device,
    _run_epoch,
)
from supervised_dataset import SupervisedIQDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune IQAutoencoder on supervised (noisy real ↔ clean re-synthesized) pairs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    data = p.add_argument_group("Data")
    data.add_argument(
        "--labels-dir", required=True,
        help="Directory containing *_labels.npz files from extract_labels.py.",
    )
    data.add_argument("--recursive", action="store_true",
                      help="Search --labels-dir recursively.")
    data.add_argument("--max-files", type=int, default=None,
                      help="Cap label files loaded (quick experiments).")
    data.add_argument("--val-fraction", type=float, default=0.15,
                      help="Fraction of pairs held out for validation.")
    data.add_argument("--amplitude-jitter", type=float, default=0.1,
                      help="Amplitude augmentation strength (0 = off).")
    data.add_argument("--no-phase-rotation", action="store_true",
                      help="Disable random-phase-rotation augmentation.")
    data.add_argument(
        "--collision-augment", type=float, default=0.10, metavar="PROB",
        help=(
            "Probability [0, 1] that each training sample has a synthetic "
            "second-aircraft signal linearly added to the noisy input. "
            "The clean target remains signal A only. "
            "0.10 = ~1 collision per 10 clean examples. 0 = disabled. "
            "Keep ≤0.15: higher values cause over-suppression at inference."
        ),
    )
    data.add_argument("--amp-ratio-min", type=float, default=0.3,
                      help="Minimum interferer amplitude as fraction of signal A.")
    data.add_argument("--amp-ratio-max", type=float, default=1.5,
                      help="Maximum interferer amplitude as fraction of signal A.")
    data.add_argument(
        "--interferer-snr-db", type=float, default=20.0, metavar="DB",
        help=(
            "SNR (dB) of AWGN added to the synthetic interferer before "
            "superimposing.  Prevents the model learning to suppress perfectly-"
            "clean structured signals, which would also suppress real primaries. "
            "Default 20 dB.  Pass a very large value (e.g. 999) to disable."
        ),
    )
    data.add_argument("--seed", type=int, default=42)

    # ── Model ───────────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model")
    mdl.add_argument(
        "--base-channels", type=int, default=64,
        help=(
            "Must match the checkpoint architecture when using --warm-start. "
            "Default 64 matches the synthetic training default."
        ),
    )
    mdl.add_argument("--depth", type=int, default=4)
    mdl.add_argument(
        "--warm-start", type=str, default=None, metavar="CKPT",
        help="Path to existing checkpoint for transfer learning (highly recommended).",
    )
    mdl.add_argument(
        "--freeze-encoder", action="store_true",
        help=(
            "Freeze the encoder for the first --freeze-epochs epochs. "
            "Good when the pre-trained encoder features are already useful."
        ),
    )
    mdl.add_argument("--freeze-epochs", type=int, default=5)

    # ── Training ────────────────────────────────────────────────────────────
    trn = p.add_argument_group("Training")
    trn.add_argument("--epochs",       type=int,   default=50)
    trn.add_argument("--batch-size",   type=int,   default=64,
                     help="Smaller than synthetic training — real pairs are precious.")
    trn.add_argument("--lr",           type=float, default=1e-4,
                     help="Lower learning rate for fine-tuning (avoid catastrophic forgetting).")
    trn.add_argument("--weight-decay", type=float, default=1e-4)
    trn.add_argument("--grad-clip",    type=float, default=1.0)
    trn.add_argument(
        "--loss-weights", type=float, nargs=3, default=[1.0, 0.5, 0.5],
        metavar=("W_IQ", "W_MAG", "W_PHASE"),
        help="Slightly higher phase weight than synthetic training — "
             "perfect targets make phase alignment more reliable.",
    )
    trn.add_argument("--num-workers", type=int, default=0)
    trn.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience in epochs (0 = disabled).",
    )

    # ── Output ──────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--ckpt-out",    type=str, default="checkpoints/best_supervised.pt")
    out.add_argument("--history-out", type=str, default=None,
                     help="Optional path for JSON training history.")
    out.add_argument("--device",      type=str, default=None,
                     help="'cpu', 'cuda', 'mps', or auto-detect.")

    return p


# ---------------------------------------------------------------------------
# Warm-start (partial weight loading, shape-safe)
# ---------------------------------------------------------------------------

def _warm_start(
    model: IQAutoencoder,
    ckpt_path: str,
    device: torch.device,
) -> None:
    """Load weights from checkpoint; skip tensors with mismatched shapes."""
    path = Path(ckpt_path)
    if not path.exists():
        print(f"[warm-start] WARNING: {path} not found — training from scratch.")
        return

    ckpt     = torch.load(path, map_location=device, weights_only=False)
    src      = ckpt.get("model_state", ckpt)
    dst      = model.state_dict()

    compatible = {k: v for k, v in src.items()
                  if k in dst and dst[k].shape == v.shape}
    shape_skip = [k for k, v in src.items()
                  if k in dst and dst[k].shape != v.shape]

    dst.update(compatible)
    model.load_state_dict(dst, strict=True)

    cfg = ckpt.get("config", {})
    print(f"[warm-start] {path.name}  "
          f"(ckpt base_channels={cfg.get('base_channels','?')})")
    print(f"             Transferred  : {len(compatible)}/{len(dst)} tensors")
    if shape_skip:
        print(f"             Shape mismatch: {len(shape_skip)} tensors (random init)")


# ---------------------------------------------------------------------------
# Encoder freeze helpers
# ---------------------------------------------------------------------------

def _set_encoder_grad(model: IQAutoencoder, requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith(("encoder", "bottleneck")):
            param.requires_grad_(requires_grad)


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _fmt(total: float, comp: Dict, elapsed: float) -> str:
    return (
        f"total={total:.5f}  "
        f"iq={comp['l_iq']:.5f}  "
        f"mag={comp['l_mag']:.5f}  "
        f"phase={comp['l_phase']:.5f}  "
        f"({elapsed:.1f}s)"
    )


def run_training(args: argparse.Namespace) -> Dict[str, List[float]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = _get_device(args.device)
    print(f"Device : {dev}")

    # ── Dataset ───────────────────────────────────────────────────────────
    full_ds = SupervisedIQDataset(
        labels_dir=args.labels_dir,
        recursive=args.recursive,
        max_files=args.max_files,
        amplitude_jitter=args.amplitude_jitter,
        phase_rotation=not args.no_phase_rotation,
        normalise=True,
        collision_augment=args.collision_augment,
        amp_ratio_min=args.amp_ratio_min,
        amp_ratio_max=args.amp_ratio_max,
        interferer_snr_db=args.interferer_snr_db,
        seed=args.seed,
    )
    print(full_ds.summary())

    if len(full_ds) < 10:
        print("WARNING: very few training pairs.  "
              "Run extract_labels.py on more .npy files first.")

    n_val   = max(1, int(len(full_ds) * args.val_fraction)) if args.val_fraction > 0 else 0
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)

    if n_val > 0:
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
        print(f"Split : {n_train:,} train  |  {n_val:,} val")
    else:
        train_ds = full_ds
        val_ds   = None

    ldr_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **ldr_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **ldr_kw) if val_ds else None

    # ── Model ─────────────────────────────────────────────────────────────
    config = AutoencoderConfig(
        seq_len=240, in_channels=2,
        base_channels=args.base_channels, depth=args.depth,
    )
    model = IQAutoencoder(config=config).to(dev)

    if args.warm_start:
        _warm_start(model, args.warm_start, dev)

    loss_fn = PhaseAwareLoss(
        w_iq=args.loss_weights[0],
        w_mag=args.loss_weights[1],
        w_phase=args.loss_weights[2],
    )

    encoder_frozen = False
    if args.freeze_encoder:
        _set_encoder_grad(model, requires_grad=False)
        encoder_frozen = True
        print(f"[freeze] Encoder frozen for first {args.freeze_epochs} epoch(s).  "
              f"Trainable: {_count_trainable(model):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100.0,
    )

    print(f"Model  : IQAutoencoder  {model.count_parameters():,} params  "
          f"{_count_trainable(model):,} trainable")
    print(f"Loss   : w_iq={args.loss_weights[0]}  "
          f"w_mag={args.loss_weights[1]}  w_phase={args.loss_weights[2]}")
    print(f"Optim  : Adam  lr={args.lr}  wd={args.weight_decay}")
    print(f"{'─' * 70}")

    # ── Checkpoint + history ──────────────────────────────────────────────
    ckpt_path = Path(args.ckpt_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    epochs_no_improve = 0

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
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1, eta_min=args.lr / 100.0,
            )
            print(f"[freeze] Encoder unfrozen at epoch {epoch}.  "
                  f"Trainable: {_count_trainable(model):,}")

        t0 = time.time()
        train_loss, train_comp = _run_epoch(
            model, train_loader, loss_fn, optimizer, dev, grad_clip=args.grad_clip,
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
                model, val_loader, loss_fn, optimizer=None, device=dev,
            )
            history["val_loss"].append(val_loss)
            for k in ("l_iq", "l_mag", "l_phase"):
                history[f"val_{k}"].append(val_comp[k])
            monitor = val_loss
            val_str = f"  |  val {_fmt(val_loss, val_comp, 0.0)}"
        else:
            monitor = train_loss
            val_str = ""

        if monitor < best_loss:
            best_loss = monitor
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "config": asdict(model.config),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": monitor,
                    "training_mode": "supervised_real",
                    "collision_augment": args.collision_augment,
                    "interferer_snr_db": args.interferer_snr_db,
                },
                ckpt_path,
            )
            val_tag = "  ✓ best"
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train {_fmt(train_loss, train_comp, time.time() - t0)}"
            f"{val_str}{val_tag}"
        )

        # ── Early stopping ────────────────────────────────────────────────
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\n[early stop] No improvement for {args.patience} epochs.  Stopping.")
            break

    print(f"{'─' * 70}")
    print(f"Training complete.  Best loss: {best_loss:.6f}")
    print(f"Checkpoint → {ckpt_path}")

    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_out).write_text(json.dumps(history, indent=2))
        print(f"History   → {args.history_out}")

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if not Path(args.labels_dir).exists():
        print(
            f"ERROR: --labels-dir '{args.labels_dir}' not found.\n"
            "Run extract_labels.py first:\n"
            "  python extract_labels.py "
            "--data-dir ~/adsb_iq_data --labels-dir ~/adsb_iq_data/labels",
            file=sys.stderr,
        )
        sys.exit(1)

    run_training(args)


if __name__ == "__main__":
    main()
