"""
evaluate.py — Visual Evaluation of the Trained IQAutoencoder
=============================================================
Loads (or auto-trains) the IQAutoencoder, generates a severe co-channel
collision test scenario, runs a single forward pass, and produces a 3×3
matplotlib visualisation that maps every stage of the denoising pipeline:

  Row 0  Time domain       — Messy input / Clean target / Reconstructed I,Q + Phase error
  Row 1  Complex plane     — Input Lissajous / Target arc / Reconstructed arc
  Row 2  Envelope & Error  — Full magnitude envelopes / Preamble zoom / Per-sample MSE

Usage
-----
  python evaluate.py                          # auto-trains if no checkpoint
  python evaluate.py --ckpt checkpoints/best.pt
  python evaluate.py --ckpt checkpoints/best.pt --snr 8 --save eval_hard.png
  python evaluate.py --train-epochs 40        # force re-train N epochs first
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sibling imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))
from generator import (
    ADSBSignalGenerator,
    SignalParams,
    FRAME_SAMPLES,
    PREAMBLE_SAMPLES,
    PREAMBLE_PULSE_INDICES,
    SAMPLE_PERIOD,
    SAMPLE_RATE,
    compute_beat_info,
)
from model import (
    AutoencoderConfig,
    IQAutoencoder,
    load_model,
    save_model,
    train,
)


# ---------------------------------------------------------------------------
# Colour palette (consistent across all panels)
# ---------------------------------------------------------------------------
C_INPUT  = "#d62728"   # red    — noisy / collision composite
C_TARGET = "#1f77b4"   # blue   — clean ground truth
C_RECON  = "#2ca02c"   # green  — model reconstruction
C_BEAT   = "#9467bd"   # purple — beat envelope / annotations
ALPHA_IN = 0.55


# ---------------------------------------------------------------------------
# 1. Model loader / auto-trainer
# ---------------------------------------------------------------------------

def _auto_train(
    ckpt_path: Path,
    epochs: int = 30,
    base_channels: int = 32,
    train_samples: int = 8192,
    seed: int = 42,
) -> None:
    """
    Train the model from scratch and save the best checkpoint.

    Called automatically when `ckpt_path` does not exist and no explicit
    checkpoint was supplied via --ckpt.
    """
    print(f"No checkpoint found at {ckpt_path}.")
    print(f"Auto-training IQAutoencoder "
          f"(base_channels={base_channels}, epochs={epochs}) …")
    t0 = time.time()
    train(
        n_epochs=epochs,
        batch_size=128,
        train_samples=train_samples,
        val_samples=1024,
        lr=1e-3,
        model_config=AutoencoderConfig(base_channels=base_channels),
        checkpoint_dir=str(ckpt_path.parent),
        seed=seed,
        include_collisions=True,
        collision_fraction=0.30,
        verbose=True,
    )
    print(f"Training finished in {time.time() - t0:.1f}s.\n")


# ---------------------------------------------------------------------------
# 2. Test scenario generator
# ---------------------------------------------------------------------------

def generate_test_scenario(
    snr_db: float = 12.0,
    f_offset_a: float = 35_000.0,
    f_offset_b: float = -20_000.0,
    amplitude_a: float = 1.0,
    amplitude_b: float = 0.60,
    initial_phase_a: float = 0.0,
    initial_phase_b_deg: float = 73.0,
    time_offset_samples: int = 7,
    receiver_dc_i: float = +0.07,
    receiver_dc_q: float = -0.05,
    phase_noise_rad: float = 0.04,
    seed: int = 99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SignalParams, SignalParams]:
    """
    Generate a severe co-channel collision test case.

    Returns
    -------
    (collision_iq, clean_a, clean_b, params_a, params_b)
    Each IQ array has shape (2, FRAME_SAMPLES).
    """
    gen = ADSBSignalGenerator(rng_seed=seed)
    phi_b = float(np.radians(initial_phase_b_deg))

    params_a = SignalParams(
        amplitude=amplitude_a,
        f_offset=f_offset_a,
        initial_phase=initial_phase_a,
        snr_db=snr_db,
    )
    params_b = SignalParams(
        amplitude=amplitude_b,
        f_offset=f_offset_b,
        initial_phase=phi_b,
        snr_db=snr_db,
    )

    collision_iq, clean_a, clean_b = gen.synthesize_collision(
        params_a, params_b,
        time_offset_samples=time_offset_samples,
        receiver_dc_i=receiver_dc_i,
        receiver_dc_q=receiver_dc_q,
    )

    # Add extra phase noise on top (severity dial)
    if phase_noise_rad > 0:
        clean_a_for_noise = gen.synthesize_clean(params_a)
        noisy_version = gen.add_impairments(
            clean_a_for_noise + clean_b,
            snr_db=snr_db - 3.0,
            dc_offset_i=receiver_dc_i,
            dc_offset_q=receiver_dc_q,
            phase_noise_rad=phase_noise_rad,
        )
        collision_iq = noisy_version

    return collision_iq, clean_a, clean_b, params_a, params_b


# ---------------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: IQAutoencoder,
    collision_iq: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Format the raw IQ array into a (1, 2, 240) tensor, run a single forward
    pass, and return the reconstruction as a (2, 240) float32 array.
    """
    x = torch.from_numpy(collision_iq).unsqueeze(0).to(device)   # (1, 2, 240)
    model.eval()
    with torch.no_grad():
        y = model(x)                         # (1, 2, 240)
    return y.squeeze(0).cpu().numpy()        # (2, 240)


# ---------------------------------------------------------------------------
# 4. Quantitative metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    collision_iq: np.ndarray,
    clean_a: np.ndarray,
    recon_iq: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a set of scalar quality metrics comparing:
      - input (collision) vs target (clean)
      - reconstruction   vs target (clean)

    Returns a dict suitable for printing and annotation.
    """
    eps = 1e-8

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    def mag(iq):
        return np.sqrt(iq[0] ** 2 + iq[1] ** 2)

    def phase_err(iq, ref):
        phi_iq  = np.arctan2(iq[1],  iq[0])
        phi_ref = np.arctan2(ref[1], ref[0])
        circ    = 1.0 - np.cos(phi_iq - phi_ref)
        weight  = mag(ref) / (mag(ref).mean() + eps)
        return float(np.sum(circ * weight) / (np.sum(weight) + eps))

    mag_in  = mag(collision_iq)
    mag_tgt = mag(clean_a)
    mag_rec = mag(recon_iq)

    mse_in  = mse(collision_iq, clean_a)
    mse_rec = mse(recon_iq,     clean_a)
    improvement = mse_in / (mse_rec + eps)

    return {
        "MSE input→target":  mse_in,
        "MSE recon→target":  mse_rec,
        "MSE improvement":   improvement,
        "MAE mag input":     float(np.mean(np.abs(mag_in  - mag_tgt))),
        "MAE mag recon":     float(np.mean(np.abs(mag_rec - mag_tgt))),
        "Phase err input":   phase_err(collision_iq, clean_a),
        "Phase err recon":   phase_err(recon_iq,     clean_a),
    }


# ---------------------------------------------------------------------------
# 5. Visualisation  (3 × 3 grid)
# ---------------------------------------------------------------------------

def plot_evaluation(
    collision_iq: np.ndarray,
    clean_a: np.ndarray,
    recon_iq: np.ndarray,
    params_a: SignalParams,
    params_b: SignalParams,
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    3 × 3 evaluation grid:

    Row 0  [ I channel waveforms ] [ Q channel waveforms ] [ Phase error / sample ]
    Row 1  [ Input constellation ] [ Target constellation] [ Reconstructed constel.]
    Row 2  [ Magnitude envelopes ] [ Preamble zoom (40 s)] [ Per-sample MSE drop  ]
    """
    sp  = SAMPLE_PERIOD
    N   = FRAME_SAMPLES
    t   = np.arange(N) * sp * 1e6           # time in µs
    eps = 1e-8

    beat = compute_beat_info(params_a, params_b)

    # ── Figure skeleton ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0e1117")

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        left=0.06, right=0.97,
        top=0.91,  bottom=0.06,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    # Dark theme helper
    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#161b22")
        ax.set_title(title, fontsize=8.5, color="white", pad=4)
        ax.set_xlabel(xlabel, fontsize=7.5, color="#aaa")
        ax.set_ylabel(ylabel, fontsize=7.5, color="#aaa")
        ax.tick_params(colors="#888", labelsize=6.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.grid(True, color="#222", lw=0.6)

    def _preamble_vline(ax):
        ax.axvline(x=PREAMBLE_SAMPLES * sp * 1e6, color="#9467bd",
                   ls=":", lw=0.8, alpha=0.6)

    fig.suptitle(
        f"IQAutoencoder Evaluation — Co-channel Collision  "
        f"│  A: {params_a.f_offset/1e3:+.0f} kHz, amp={params_a.amplitude:.2f}  "
        f"│  B: {params_b.f_offset/1e3:+.0f} kHz, amp={params_b.amplitude:.2f}  "
        f"│  f_beat={beat['freq_separation_hz']/1e3:.0f} kHz  "
        f"│  SNR={params_a.snr_db:.0f} dB\n"
        f"MSE: {metrics['MSE input→target']:.4f} → {metrics['MSE recon→target']:.4f}  "
        f"({metrics['MSE improvement']:.1f}× improvement)  "
        f"│  Phase err: {metrics['Phase err input']:.3f} → {metrics['Phase err recon']:.3f}",
        fontsize=9, color="white", y=0.975,
    )

    # ── Helper: mag / phase ──────────────────────────────────────────────────
    def _mag(iq):
        return np.sqrt(iq[0] ** 2 + iq[1] ** 2)

    def _phi(iq):
        return np.arctan2(iq[1], iq[0])

    def _circ_err(iq, ref):
        return 1.0 - np.cos(_phi(iq) - _phi(ref))

    mag_in  = _mag(collision_iq)
    mag_tgt = _mag(clean_a)
    mag_rec = _mag(recon_iq)

    on_mask = mag_tgt > 0.05   # on-state samples (meaningful target signal)

    # ════════════════════════════════════════════════════════════════════════
    # Row 0 — Time-domain waveforms
    # ════════════════════════════════════════════════════════════════════════

    for col, (ch_idx, ch_name) in enumerate([(0, "I (In-Phase)"), (1, "Q (Quadrature)")]):
        ax = axes[0][col]
        ax.plot(t, collision_iq[ch_idx], color=C_INPUT,  lw=0.8, alpha=ALPHA_IN, label="Input (collision)")
        ax.plot(t, clean_a[ch_idx],      color=C_TARGET, lw=1.6, label="Clean target")
        ax.plot(t, recon_iq[ch_idx],     color=C_RECON,  lw=1.2, ls="--", label="Reconstructed")
        _preamble_vline(ax)
        _style(ax, ch_name, "Time (µs)", "Amplitude")
        ax.legend(fontsize=6.5, loc="upper right",
                  facecolor="#222", labelcolor="white", edgecolor="#555")

    # Phase error per sample over time
    ax = axes[0][2]
    err_in  = _circ_err(collision_iq, clean_a)
    err_rec = _circ_err(recon_iq,     clean_a)
    weight  = mag_tgt / (mag_tgt.mean() + eps)

    ax.fill_between(t, err_in * weight,  color=C_INPUT,  alpha=0.35, label="Input phase error")
    ax.fill_between(t, err_rec * weight, color=C_RECON,  alpha=0.55, label="Recon phase error")
    ax.plot(t, err_in * weight,  color=C_INPUT,  lw=0.7, alpha=0.7)
    ax.plot(t, err_rec * weight, color=C_RECON,  lw=1.0)
    _preamble_vline(ax)
    _style(ax, "Circular Phase Error  (1 − cos Δφ) × w_signal",
           "Time (µs)", "Weighted phase error")
    ax.legend(fontsize=6.5, facecolor="#222", labelcolor="white", edgecolor="#555")

    # ════════════════════════════════════════════════════════════════════════
    # Row 1 — IQ Constellations
    # ════════════════════════════════════════════════════════════════════════

    theta_circ = np.linspace(0, 2 * np.pi, 300)

    constellation_data = [
        (collision_iq, C_INPUT,  "Input — Collision Lissajous"),
        (clean_a,      C_TARGET, "Target — Clean Primary Aircraft"),
        (recon_iq,     C_RECON,  "Reconstructed — Model Output"),
    ]
    for col, (iq, colour, title) in enumerate(constellation_data):
        ax = axes[1][col]
        ax.plot(np.cos(theta_circ), np.sin(theta_circ),
                color="white", lw=0.5, alpha=0.15, zorder=1)

        # On-state samples coloured by time index (shows rotation direction)
        mask = _mag(iq) > 0.05
        if mask.sum() > 1:
            sc = ax.scatter(iq[0, mask], iq[1, mask],
                            c=np.where(mask)[0], cmap="plasma",
                            s=8, zorder=3, alpha=0.85,
                            vmin=0, vmax=N)
        # Off-state cluster
        ax.scatter(iq[0, ~mask], iq[1, ~mask],
                   s=3, color="#444", alpha=0.4, zorder=2)

        _style(ax, title, "I", "Q")
        ax.set_aspect("equal", adjustable="box")
        # Overlay target arc on recon panel for direct comparison
        if col == 2 and clean_a is not None:
            tgt_on = _mag(clean_a) > 0.05
            ax.scatter(clean_a[0, tgt_on], clean_a[1, tgt_on],
                       s=12, color=C_TARGET, alpha=0.35, zorder=4,
                       label="Target arc")
            ax.legend(fontsize=6, facecolor="#222", labelcolor="white",
                      edgecolor="#555")

    # Shared colour bar for constellation panels
    cbar = fig.colorbar(sc, ax=axes[1], orientation="vertical",
                        fraction=0.01, pad=0.02)
    cbar.set_label("Sample index", fontsize=7, color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # ════════════════════════════════════════════════════════════════════════
    # Row 2 — Magnitude & Error Analysis
    # ════════════════════════════════════════════════════════════════════════

    # [2,0] Full magnitude envelopes — beating, pulse recovery
    ax = axes[2][0]
    ax.plot(t, mag_in,  color=C_INPUT,  lw=0.9, alpha=ALPHA_IN, label="Input |IQ|")
    ax.plot(t, mag_tgt, color=C_TARGET, lw=1.8, label="Target |IQ|")
    ax.plot(t, mag_rec, color=C_RECON,  lw=1.2, ls="--", label="Recon |IQ|")
    ax.axhline(y=beat["envelope_max"], color=C_BEAT, ls="--", lw=0.8, alpha=0.5,
               label=f"Beat max {beat['envelope_max']:.2f}")
    ax.axhline(y=beat["envelope_min"], color=C_BEAT, ls=":",  lw=0.8, alpha=0.5,
               label=f"Beat min {beat['envelope_min']:.2f}")
    _preamble_vline(ax)
    _style(ax,
           f"Magnitude Envelopes — f_beat={beat['freq_separation_hz']/1e3:.0f} kHz  "
           f"({beat['beats_per_frame']:.1f} cycles/frame)",
           "Time (µs)", "|I + jQ|")
    ax.legend(fontsize=6, facecolor="#222", labelcolor="white", edgecolor="#555",
              ncol=2)

    # [2,1] Preamble zoom — verify pulse position recovery
    ZOOM = 40   # first 40 samples = 20 µs
    ax = axes[2][1]
    t_z = t[:ZOOM]
    ax.plot(t_z, mag_in[:ZOOM],  color=C_INPUT,  lw=0.9, alpha=ALPHA_IN, label="Input")
    ax.plot(t_z, mag_tgt[:ZOOM], color=C_TARGET, lw=2.0, label="Target")
    ax.plot(t_z, mag_rec[:ZOOM], color=C_RECON,  lw=1.3, ls="--", label="Recon")
    # Mark canonical preamble pulse positions
    for idx in PREAMBLE_PULSE_INDICES:
        ax.axvline(x=idx * sp * 1e6, color="yellow", lw=0.7, ls=":", alpha=0.6)
    ax.axvline(x=PREAMBLE_SAMPLES * sp * 1e6, color="#9467bd",
               lw=0.9, ls="--", alpha=0.7, label="Preamble end")
    _style(ax, "Preamble Zoom — First 20 µs\n(yellow = canonical pulse positions)",
           "Time (µs)", "|IQ|")
    ax.legend(fontsize=6, facecolor="#222", labelcolor="white", edgecolor="#555")

    # [2,2] Per-sample MSE drop across the frame
    ax = axes[2][2]
    mse_in_per_sample  = np.mean((collision_iq - clean_a) ** 2, axis=0)
    mse_rec_per_sample = np.mean((recon_iq     - clean_a) ** 2, axis=0)
    # Smooth with a small running window so spikes don't dominate visually
    kernel = np.ones(5) / 5
    mse_in_sm  = np.convolve(mse_in_per_sample,  kernel, mode="same")
    mse_rec_sm = np.convolve(mse_rec_per_sample, kernel, mode="same")

    ax.fill_between(t, mse_in_sm,  color=C_INPUT, alpha=0.35,
                    label=f"Input MSE  (mean={mse_in_per_sample.mean():.4f})")
    ax.fill_between(t, mse_rec_sm, color=C_RECON, alpha=0.55,
                    label=f"Recon MSE  (mean={mse_rec_per_sample.mean():.4f})")
    ax.plot(t, mse_in_sm,  color=C_INPUT, lw=0.7, alpha=0.7)
    ax.plot(t, mse_rec_sm, color=C_RECON, lw=1.0)
    _preamble_vline(ax)
    _style(ax, "Per-sample MSE  (5-sample smoothed)\nRed = input error,  Green = reconstruction error",
           "Time (µs)", "MSE")
    ax.legend(fontsize=6.5, facecolor="#222", labelcolor="white", edgecolor="#555")

    # ── Finalise ─────────────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Evaluation figure saved → {save_path}")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# 6. Metrics summary
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict[str, float], params_a: SignalParams,
                  params_b: SignalParams) -> None:
    beat = compute_beat_info(params_a, params_b)
    sep  = "─" * 54

    print(f"\n{sep}")
    print("  EVALUATION METRICS")
    print(sep)
    print(f"  Test scenario:")
    print(f"    Aircraft A : f={params_a.f_offset/1e3:+.1f} kHz, amp={params_a.amplitude:.2f}, φ₀={params_a.initial_phase:.2f} rad")
    print(f"    Aircraft B : f={params_b.f_offset/1e3:+.1f} kHz, amp={params_b.amplitude:.2f}")
    print(f"    SNR        : {params_a.snr_db:.1f} dB  (challenging)")
    print(f"    f_beat     : {beat['freq_separation_hz']/1e3:.1f} kHz  "
          f"→ {beat['beats_per_frame']:.1f} cycles/frame")
    print(f"    Mod. depth : {beat['modulation_depth']*100:.0f} %")
    print(sep)
    print(f"  IQ channel MSE:")
    print(f"    Input  → target :  {metrics['MSE input→target']:.6f}")
    print(f"    Recon  → target :  {metrics['MSE recon→target']:.6f}")
    print(f"    Improvement     :  {metrics['MSE improvement']:.2f}×")
    print(f"  Magnitude MAE:")
    print(f"    Input  → target :  {metrics['MAE mag input']:.6f}")
    print(f"    Recon  → target :  {metrics['MAE mag recon']:.6f}")
    print(f"  Circular phase error  (weighted, on-state samples):")
    print(f"    Input  → target :  {metrics['Phase err input']:.6f}")
    print(f"    Recon  → target :  {metrics['Phase err recon']:.6f}")
    print(sep)


# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained IQAutoencoder on a collision scenario."
    )
    parser.add_argument("--ckpt",          type=str,   default="checkpoints/best.pt",
                        help="Path to model checkpoint (default: checkpoints/best.pt)")
    parser.add_argument("--train-epochs",  type=int,   default=30,
                        help="Epochs to train if no checkpoint found (default: 30)")
    parser.add_argument("--base-channels", type=int,   default=32,
                        help="base_channels for auto-train (default: 32)")
    parser.add_argument("--snr",           type=float, default=12.0,
                        help="SNR in dB for the test scenario (default: 12)")
    parser.add_argument("--f-a",           type=float, default=35_000.0,
                        help="Frequency offset of aircraft A in Hz (default: +35000)")
    parser.add_argument("--f-b",           type=float, default=-20_000.0,
                        help="Frequency offset of aircraft B in Hz (default: -20000)")
    parser.add_argument("--save",          type=str,   default="evaluation_results.png",
                        help="Output figure filename (default: evaluation_results.png)")
    parser.add_argument("--no-show",       action="store_true",
                        help="Skip plt.show() — useful in headless environments")
    parser.add_argument("--device",        type=str,   default=None,
                        help="Force device: cpu / cuda / mps")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)

    # ── Step 1: Load or auto-train ────────────────────────────────────────
    if not ckpt_path.exists():
        _auto_train(
            ckpt_path,
            epochs=args.train_epochs,
            base_channels=args.base_channels,
        )

    print(f"Loading model from {ckpt_path} …")
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    model = load_model(ckpt_path, device=str(device))
    cfg   = model.config
    print(f"  Model   : base_channels={cfg.base_channels}  depth={cfg.depth}  "
          f"params={model.count_parameters():,}")
    print(f"  Device  : {device}")

    # ── Step 2: Generate test scenario ───────────────────────────────────
    print("\nGenerating test scenario …")
    collision_iq, clean_a, clean_b, params_a, params_b = generate_test_scenario(
        snr_db=args.snr,
        f_offset_a=args.f_a,
        f_offset_b=args.f_b,
    )
    print(f"  Collision shape : {collision_iq.shape}  dtype={collision_iq.dtype}")

    # ── Step 3: Inference ─────────────────────────────────────────────────
    print("Running forward pass …")
    t0    = time.time()
    recon = run_inference(model, collision_iq, device)
    ms    = (time.time() - t0) * 1000
    print(f"  Inference time  : {ms:.2f} ms  →  {recon.shape}")

    # ── Step 4: Metrics ───────────────────────────────────────────────────
    metrics = compute_metrics(collision_iq, clean_a, recon)
    print_metrics(metrics, params_a, params_b)

    # ── Step 5: Visualise ─────────────────────────────────────────────────
    print("\nRendering evaluation figure …")
    save_path = str(Path(__file__).parent / args.save) if args.save else None
    plot_evaluation(
        collision_iq, clean_a, recon,
        params_a, params_b,
        metrics,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
