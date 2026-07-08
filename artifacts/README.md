# Processed outputs (local only)

Model-processed `.bin` files from `live_bridge.py` and evaluation runs.
Gitignored — regenerate with:

```bash
python live_bridge.py --ckpt checkpoints/best_supervised_v4.pt \
    --blend 0.05 --input data/captures/test_capture_36db.bin \
    --output artifacts/test_blend005.bin
```
