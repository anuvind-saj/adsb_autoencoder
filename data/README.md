# Data (local only — not committed to GitHub)

Place RTL-SDR captures here for evaluation and reproduction.

## Benchmark captures used in the paper

| File | Description |
|---|---|
| `captures/adsb_capture.bin` | In-distribution benchmark (480 MB) |
| `captures/test_capture_36db.bin` | OOD holdout, 1090 MHz, 36 dB (120 MB) |
| `captures/capture_36db_20260707_145107.bin` | Fresh Pi capture (120 MB) |

## Training corpus

`adsb_iq_data/` — raw ML collection bursts (406 GB locally; never pushed to Git).

Obtain captures with:

```bash
rtl_sdr -f 1090000000 -s 2000000 -g 36 -n 60000000 capture.bin
```
