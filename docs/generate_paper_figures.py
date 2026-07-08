#!/usr/bin/env python3
"""
generate_paper_figures.py — Publication figures for PAPER.md / paper.tex

Outputs (figures/):
  fig_pipeline.png       System block diagram
  fig_ppm_grid.png       PPM sample grid at 2 MSPS
  fig_collision.png      Synthetic collision IQ constellation (best.pt)
  fig_results_bars.png   Decode frame counts: raw vs full model vs blend
  fig_blend_sweep.png    Alpha sweep on test_capture_36db.bin
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch

ROOT = Path(__file__).parent          # docs/
REPO_ROOT = ROOT.parent
FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(REPO_ROOT))

from evaluate import generate_test_scenario
from generator import (
    FRAME_SAMPLES,
    PREAMBLE_PULSE_INDICES,
    PREAMBLE_SAMPLES,
    SAMPLES_PER_BIT,
    SAMPLE_PERIOD,
)
from model import load_model
from compare_decodings import decode_iq, load_iq_normalized

# Paper palette
C_RAW = "#4c72b0"
C_MODEL = "#c44e52"
C_BLEND = "#55a868"
C_INPUT = "#d62728"
C_TARGET = "#1f77b4"
C_RECON = "#2ca02c"


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {out}")
    return out


def fig_pipeline() -> None:
    """Block diagram: capture → labels → train → bridge → decode."""
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.2, 1.0, "RTL-SDR\nCapture", "#e8f4fc"),
        (2.0, 1.0, "extract_labels.py\nCRC + re-synth", "#fff3cd"),
        (4.0, 1.0, "train_supervised.py\nIQAutoencoder", "#d4edda"),
        (6.2, 1.0, "live_bridge.py\nOLA + blend", "#f8d7da"),
        (8.4, 1.0, "dump1090 /\ncompare_decodings", "#e2d9f3"),
    ]
    for x, y, text, color in boxes:
        box = FancyBboxPatch(
            (x, y), 1.55, 1.1,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 0.775, y + 0.55, text, ha="center", va="center",
                fontsize=9, fontweight="bold")

    for x0, x1 in [(1.75, 2.0), (3.55, 4.0), (5.75, 6.2), (7.95, 8.4)]:
        ax.annotate(
            "", xy=(x1, 1.55), xytext=(x0, 1.55),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5),
        )

    ax.text(5.5, 2.55, "Evaluation metric: CRC-valid Mode S frame count",
            ha="center", fontsize=10, style="italic", color="#444")
    ax.set_title("End-to-end ADS-B IQ autoencoder pipeline", fontsize=12, fontweight="bold", pad=8)
    _save(fig, "fig_pipeline.png")


def fig_ppm_grid() -> None:
    """Schematic: 240-sample frame aligned to PPM bit halves at 2 MSPS."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), gridspec_kw={"height_ratios": [1, 2.2]})

    n = FRAME_SAMPLES
    t_us = np.arange(n) * SAMPLE_PERIOD * 1e6

    # Envelope schematic (preamble + first 8 bits)
    env = np.zeros(n)
    for idx in PREAMBLE_PULSE_INDICES:
        env[idx] = 1.0
    for bit in range(8):
        base = PREAMBLE_SAMPLES + bit * SAMPLES_PER_BIT
        if bit % 2 == 0:
            env[base] = 1.0
        else:
            env[base + 1] = 1.0

    ax0 = axes[0]
    ax0.fill_between(t_us, 0, env, step="mid", alpha=0.35, color=C_TARGET)
    ax0.plot(t_us, env, drawstyle="steps-mid", color=C_TARGET, lw=1.5)
    ax0.axvline(PREAMBLE_SAMPLES * SAMPLE_PERIOD * 1e6, color="purple", ls=":", lw=1.2)
    ax0.set_xlim(0, 12)
    ax0.set_ylim(-0.1, 1.35)
    ax0.set_ylabel("OOK envelope")
    ax0.set_title("Mode S frame: 240 samples = 120 µs at 2.0 MSPS (0.5 µs / sample)",
                  fontsize=11, fontweight="bold")
    ax0.text(4.0, 1.15, "Preamble (16 samples)", fontsize=8, color="purple")
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    colors = []
    for i in range(n):
        if i in PREAMBLE_PULSE_INDICES:
            colors.append("#9467bd")
        elif i < PREAMBLE_SAMPLES:
            colors.append("#dddddd")
        elif (i - PREAMBLE_SAMPLES) % 2 == 0:
            colors.append("#aec7e8")
        else:
            colors.append("#ffbb78")
    ax1.bar(t_us[:48], np.ones(48), width=SAMPLE_PERIOD * 1e6 * 0.9, color=colors[:48], edgecolor="none")
    ax1.set_xlim(0, 12)
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel("Sample index")
    ax1.set_yticks([])
    ax1.set_title("First 48 samples: purple = preamble pulses; blue/orange = bit halves",
                  fontsize=10)
    legend_patches = [
        mpatches.Patch(color="#9467bd", label="Preamble pulse"),
        mpatches.Patch(color="#aec7e8", label="Bit=1 (1st half)"),
        mpatches.Patch(color="#ffbb78", label="Bit=0 (2nd half)"),
    ]
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_ppm_grid.png")


def fig_collision(ckpt: Path = REPO_ROOT / "checkpoints" / "best.pt") -> None:
    """IQ constellation: collision input → target A → v4/synthetic reconstruction."""
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_model(ckpt, device=str(device))
    model.eval()

    collision, clean_a, _, _, _ = generate_test_scenario(seed=7)

    with torch.no_grad():
        x = torch.from_numpy(collision).unsqueeze(0).to(device)
        recon = model(x).squeeze(0).cpu().numpy()

    def _constellation(ax, iq, title, color, alpha=0.7):
        on = iq[0] ** 2 + iq[1] ** 2 > 0.01
        theta = np.linspace(0, 2 * np.pi, 128)
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.6, alpha=0.25)
        ax.scatter(iq[0, on], iq[1, on], s=12, c=color, alpha=alpha, edgecolors="none")
        ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.2)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    _constellation(axes[0], collision, "Collision (2 aircraft)", C_INPUT)
    _constellation(axes[1], clean_a, "Target: aircraft A", C_TARGET)
    _constellation(axes[2], recon, "Reconstructed (model)", C_RECON)
    fig.suptitle(
        "Synthetic co-channel collision: Lissajous mixture → single-source recovery",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_collision.png")


def fig_results_bars() -> None:
    """Grouped bars: raw vs α=1 vs α=0.05 for three captures."""
    captures = [
        "adsb_capture\n(in-dist.)",
        "test_36db\n(OOD)",
        "Pi capture\nJul 2026",
    ]
    raw = [1589, 314, 217]
    model = [83, 1, None]   # v4 α=1; Pi full-model not benchmarked
    blend = [1589, 295, 194]

    x = np.arange(len(captures))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - w, raw, w, label="Raw", color=C_RAW)
    ax.bar(x, [m if m is not None else 0 for m in model], w,
           label="v4 full model (α=1)", color=C_MODEL)
    ax.bar(x + w, blend, w, label="v4 blend (α=0.05)", color=C_BLEND)
    for i, m in enumerate(model):
        if m is None:
            ax.text(x[i], 5, "n/a", ha="center", fontsize=8, color="#666")
    ax.set_ylabel("CRC-valid frames")
    ax.set_xticks(x)
    ax.set_xticklabels(captures, fontsize=9)
    ax.set_title("Decode-gated results: full model fails; 5% blend preserves decodability",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_results_bars.png")


def fig_blend_sweep(
    raw_path: Path = REPO_ROOT / "data" / "captures" / "test_capture_36db.bin",
    model_path: Path = REPO_ROOT / "artifacts" / "test_v4_36db_clean.bin",
) -> None:
    """Alpha vs frame count on OOD capture."""
    if not raw_path.exists() or not model_path.exists():
        print(f"  skip blend sweep — missing {raw_path.name} or {model_path.name}")
        return

    alphas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    raw_i, raw_q = load_iq_normalized(raw_path)
    mod_i, mod_q = load_iq_normalized(model_path)
    n = min(len(raw_i), len(mod_i))
    raw_i, raw_q = raw_i[:n], raw_q[:n]
    mod_i, mod_q = mod_i[:n], mod_q[:n]

    frames = []
    for a in alphas:
        bi = a * mod_i + (1 - a) * raw_i
        bq = a * mod_q + (1 - a) * raw_q
        msgs, _, _ = decode_iq(bi, bq, verbose=False)
        frames.append(len(msgs))
        print(f"    α={a:.2f} → {len(msgs)} frames")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(alphas, frames, "o-", color=C_BLEND, lw=2, markersize=7)
    ax.axhline(314, color=C_RAW, ls="--", lw=1.2, alpha=0.7, label="Raw baseline (314)")
    ax.axvline(0.05, color="#888", ls=":", lw=1.2, label="Deploy α=0.05")
    ax.set_xlabel("α  (model fraction)")
    ax.set_ylabel("CRC-valid frames")
    ax.set_title("Blend sweep on test_capture_36db.bin (OOD)", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_blend_sweep.png")


def main() -> None:
    print("Generating paper figures …")
    fig_pipeline()
    fig_ppm_grid()
    fig_collision()
    fig_results_bars()
    fig_blend_sweep()
    print("Done.")


if __name__ == "__main__":
    main()
