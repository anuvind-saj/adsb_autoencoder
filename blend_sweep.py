"""
blend_sweep.py — Sweep raw/model blend ratios and report decode frame counts.

Blends in normalized float IQ space:
    blended = alpha * model + (1 - alpha) * raw

alpha = 0   → pure raw (sanity check: should match raw decode)
alpha = 1   → pure model output

Typical usage
-------------
# Sweep offline (model output file must already exist):
python blend_sweep.py test_capture_36db.bin test_v4_36db_clean.bin

# Same mix during inference (no separate sweep needed):
python live_bridge.py --ckpt checkpoints/best_supervised_v4.pt \\
    --blend 0.05 --input adsb_capture.bin --output clean_blend005.bin

python blend_sweep.py adsb_capture.bin test_v4_capture_clean.bin \\
    --alphas 0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5

# Write the best-alpha blend to disk for inspection:
python blend_sweep.py test_capture_36db.bin test_v4_36db_clean.bin \\
    --write-best blend_best.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from compare_decodings import decode_iq, iq_normalized_to_bytes, load_iq_normalized


def _parse_alphas(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty --alphas list")
    for v in vals:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {v}")
    return sorted(set(vals))


def _blend(
    raw_i: np.ndarray,
    raw_q: np.ndarray,
    model_i: np.ndarray,
    model_q: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    blend_i = alpha * model_i + (1.0 - alpha) * raw_i
    blend_q = alpha * model_q + (1.0 - alpha) * raw_q
    return blend_i.astype(np.float32), blend_q.astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep alpha blends between raw and model-clean IQ captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("raw", help="Raw RTL-SDR uint8 IQ capture.")
    p.add_argument("model", help="Model-processed uint8 IQ from live_bridge.py.")
    p.add_argument(
        "--alphas",
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated blend weights (alpha=model fraction).",
    )
    p.add_argument(
        "--write-best",
        metavar="FILE",
        default=None,
        help="Write uint8 IQ for the alpha with the most valid frames.",
    )
    p.add_argument("--thr-percentile", type=float, default=90.0)
    p.add_argument("--thr-mult", type=float, default=3.5)
    p.add_argument("--thr-min", type=float, default=0.10)
    p.add_argument("--low-ratio", type=float, default=3.0)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    raw_path = Path(args.raw)
    model_path = Path(args.model)
    for path in (raw_path, model_path):
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    alphas = _parse_alphas(args.alphas)
    decoder_kw = dict(
        thr_percentile=args.thr_percentile,
        thr_multiplier=args.thr_mult,
        thr_min=args.thr_min,
        low_ratio=args.low_ratio,
        show_n=0,
        verbose=False,
    )

    print(f"Loading raw   : {raw_path.name}  ({raw_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Loading model : {model_path.name}  ({model_path.stat().st_size / 1e6:.1f} MB)")

    raw_i, raw_q = load_iq_normalized(raw_path)
    model_i, model_q = load_iq_normalized(model_path)
    n = min(len(raw_i), len(model_i))
    if len(raw_i) != len(model_i):
        print(
            f"WARNING: length mismatch raw={len(raw_i):,} model={len(model_i):,} "
            f"— using first {n:,} samples.",
            file=sys.stderr,
        )
        raw_i, raw_q = raw_i[:n], raw_q[:n]
        model_i, model_q = model_i[:n], model_q[:n]

    raw_msgs, raw_icao, _ = decode_iq(raw_i, raw_q, **decoder_kw)
    raw_frames = len(raw_msgs)
    raw_aircraft = len(raw_icao)

    print(f"\nRaw baseline: {raw_frames} frames, {raw_aircraft} aircraft\n")

    bar = "=" * 72
    print(bar)
    print(f"  {'alpha':>6}  {'model%':>7}  {'Frames':>7}  {'Aircraft':>8}  "
          f"{'Recovery':>9}  {'Candidates':>11}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*11}")

    best_alpha = 0.0
    best_frames = -1
    best_aircraft = -1
    best_i: np.ndarray | None = None
    best_q: np.ndarray | None = None

    for alpha in alphas:
        if alpha == 0.0:
            frames, aircraft = raw_frames, raw_aircraft
            candidates = 0
        else:
            bi, bq = _blend(raw_i, raw_q, model_i, model_q, alpha)
            msgs, icao, stats = decode_iq(bi, bq, **decoder_kw)
            frames = len(msgs)
            aircraft = len(icao)
            candidates = stats["candidates"]

        recovery = 100.0 * frames / max(raw_frames, 1)
        print(
            f"  {alpha:>6.2f}  {100 * alpha:>6.1f}%  {frames:>7,}  {aircraft:>8,}  "
            f"{recovery:>8.1f}%  {candidates:>11,}"
        )

        if frames > best_frames or (frames == best_frames and aircraft > best_aircraft):
            best_alpha = alpha
            best_frames = frames
            best_aircraft = aircraft
            if alpha == 0.0:
                best_i, best_q = raw_i.copy(), raw_q.copy()
            else:
                best_i, best_q = _blend(raw_i, raw_q, model_i, model_q, alpha)

    print(bar)
    print(
        f"\nBest: alpha={best_alpha:.2f}  →  {best_frames} frames, "
        f"{best_aircraft} aircraft  "
        f"({100.0 * best_frames / max(raw_frames, 1):.1f}% of raw)"
    )

    if args.write_best and best_i is not None and best_q is not None:
        out_path = Path(args.write_best)
        out_path.write_bytes(iq_normalized_to_bytes(best_i, best_q))
        print(f"Wrote best blend → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
