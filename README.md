# Phase-Aware IQ Autoencoding for ADS-B Mode S

Phase-aware 1D U-Net autoencoder for IQ-domain denoising and blind source separation of 1090 MHz ADS-B Mode S signals. Trained on RTL-SDR captures; evaluated via downstream CRC decode (`dump1090`).

**Author:** Anuvind Saj  
**Report:** [docs/PAPER.md](docs/PAPER.md) · [docs/paper.pdf](docs/paper.pdf)  
**Extended notes:** [docs/WHITEPAPER_BLUEPRINT.md](docs/WHITEPAPER_BLUEPRINT.md)

---

## Results (summary)

| Configuration | OOD capture (`test_capture_36db.bin`) |
|---|---|
| Raw | 314 frames |
| v4 full model (α=1) | 1 frame |
| **v4 blend (α=0.05)** | **295 frames (93.9%)** |

Experimental phase closed July 2026. Full analysis in the technical report.

---

## Repository layout

```
adsb_autoencoder/
├── README.md                 # this file
├── model.py, generator.py, … # core Python (run from repo root)
├── live_bridge.py            # streaming inference + --blend
├── compare_decodings.py      # CRC frame-count evaluation
├── blend_sweep.py            # offline α sweep
├── checkpoints/              # .pt weights (gitignored — see below)
├── data/
│   ├── captures/             # raw .bin captures (gitignored)
│   └── adsb_iq_data/         # training corpus (gitignored, local only)
├── artifacts/                # processed .bin outputs (gitignored)
└── docs/
    ├── PAPER.md, paper.pdf   # technical report
    ├── paper.tex             # LaTeX source
    ├── figures/              # paper figures
    └── generate_paper_figures.py
```

---

## Quick start

### Dependencies

```bash
# Conda (recommended on Mac)
conda create -n adsb python=3.11 pytorch numpy matplotlib -c pytorch
conda activate adsb
pip install -r requirements_rpi.txt   # CPU torch index; works on Mac too
```

### Evaluate a capture

Place captures in `data/captures/`, then:

```bash
python compare_decodings.py data/captures/test_capture_36db.bin
```

### Deployable blend (recommended)

```bash
python live_bridge.py \
    --ckpt checkpoints/best_supervised_v4.pt \
    --blend 0.05 \
    --input data/captures/test_capture_36db.bin \
    --output artifacts/test_blend005.bin

python compare_decodings.py \
    data/captures/test_capture_36db.bin \
    artifacts/test_blend005.bin
```

### Rebuild paper PDF

```bash
cd docs && make pdf
```

---

## Checkpoints & data (not on GitHub)

| Asset | Location | Notes |
|---|---|---|
| **Checkpoints** | `checkpoints/` | Copy manually or attach to GitHub Release |
| **Raw captures** | `data/captures/` | See [data/README.md](data/README.md) |
| **Training corpus** | `data/adsb_iq_data/` | 406 GB locally; not published |

Recommended deployment checkpoint: `checkpoints/best_supervised_v4.pt`

---

## Deploy to Raspberry Pi

```bash
./deploy.sh --push --install-deps
```

Syncs Python source and `docs/` only (no `.bin`, `.pt`, or large data).

---

## Citation

```bibtex
@techreport{saj2026adsb,
  author      = {Anuvind Saj},
  title       = {Phase-Aware IQ Autoencoding for {ADS-B} Mode S:
                 Synthetic Blind Source Separation, Real-World Constraints,
                 and Conservative Deployment via Model--Raw Blending},
  year        = {2026},
  month       = jul,
  institution = {Technical Report},
  note        = {Code: https://github.com/YOUR_USERNAME/adsb_autoencoder}
}
```

Replace the GitHub URL after publishing.

---

## License

Add a license before public release (e.g. MIT for code, CC-BY for report).
