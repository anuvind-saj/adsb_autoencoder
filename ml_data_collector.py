"""
ml_data_collector.py — ML Training Data Collector for ADS-B IQ
==============================================================

Captures raw IQ bursts from an RTL-SDR optimised for training the
IQAutoencoder, then uploads each burst to MinIO on the central Pi.

Key differences from the general-purpose adsb_iq_sample_collection/collector.py
----------------------------------------------------------------------------------
1. GAIN ROTATION — cycles through a configurable list of gain values across
   consecutive bursts.  This is the single most important improvement for
   training quality: the model must see weak signals (low gain → low SNR) as
   well as strong signals.  Recommended gains: [28, 32, 36, 40, 44] dB.

2. LONG BURSTS — default 30 s per burst (vs 5 s in the general collector).
   More samples per file → more decoded frames per extraction run →
   fewer file-open/upload overheads.

3. CONTINUOUS COLLECTION — no inter-interval idle gap.  The script captures
   back-to-back with no scheduled sleep, maximising duty cycle.

4. GAIN IN FILENAME — gain is embedded in the filename
   (<antenna>_<timestamp>_g<gain>.npy) so the training pipeline can filter
   or weight by gain level.

5. CENTRAL MINIO UPLOAD — uploads directly to MinIO on rpi-master which is
   expected to have a 1 TB SSD attached.

------------------------------------------------------------------------------
QUICK START
------------------------------------------------------------------------------

  # Install dependencies (inside your venv):
  pip install numpy minio

  # Run interactively to verify:
  python ml_data_collector.py \\
      --antenna-id rpi_west \\
      --minio-endpoint rpi-master.local:9000 \\
      --output-dir /mnt/ssd/adsb_ml_data \\
      --burst-sec 30 \\
      --gains 28 32 36 40 44 \\
      --verbose

------------------------------------------------------------------------------
SYSTEMD SERVICE SETUP
------------------------------------------------------------------------------

1. MOUNT THE 1 TB SSD ON rpi-master
   ------------------------------------
   Find the SSD device:
     lsblk
     # Look for your SSD, e.g. /dev/sda (likely ~1TB)

   Create a partition and filesystem (first time only):
     sudo fdisk /dev/sda        # create a new partition (n → p → 1 → default → w)
     sudo mkfs.ext4 /dev/sda1

   Create the mount point:
     sudo mkdir -p /mnt/ssd

   Get the UUID for persistent mounting:
     sudo blkid /dev/sda1
     # Note the UUID, e.g. UUID="a1b2c3d4-..."

   Add to /etc/fstab for auto-mount on boot:
     echo 'UUID=<your-uuid>  /mnt/ssd  ext4  defaults,noatime  0  2' | sudo tee -a /etc/fstab
     sudo mount -a           # mount now without rebooting
     df -h /mnt/ssd          # verify ~932 GB free

   Set permissions for your user:
     sudo chown -R asaj:asaj /mnt/ssd
     mkdir -p /mnt/ssd/adsb_ml_data

2. POINT MINIO AT THE SSD (on rpi-master)
   -----------------------------------------
   If MinIO is already running with data in the default directory, migrate it:

     sudo systemctl stop minio

     # Option A: Move existing data to SSD (keeps history)
     sudo mv /home/pi/adsb_iq_minio_data /mnt/ssd/minio_data

     # Option B: Start fresh on SSD (faster)
     sudo mkdir -p /mnt/ssd/minio_data

   Edit the MinIO service environment file:
     sudo nano /etc/default/minio
     # Change:   MINIO_VOLUMES="/home/pi/adsb_iq_minio_data"
     # To:       MINIO_VOLUMES="/mnt/ssd/minio_data"

   Restart MinIO:
     sudo systemctl start minio
     sudo systemctl status minio

3. DEPLOY ml_data_collector.py TO THE COLLECTOR Pi(s)
   -------------------------------------------------------
   From your Mac (using the existing deploy.sh, or manually):
     rsync -av ml_data_collector.py pi@rpi-west.local:~/workspace/adsb_autoencoder/

   Or add it to deploy.sh so it syncs automatically with the rest of the project.

4. INSTALL DEPENDENCIES ON THE COLLECTOR Pi
   -------------------------------------------
   On each collector Pi (rpi_west, rpi_east):
     cd ~/workspace/adsb_autoencoder
     source .venv/bin/activate
     pip install minio   # numpy should already be installed

5. CREATE THE SYSTEMD SERVICE ON THE COLLECTOR Pi
   --------------------------------------------------
   SSH into the collector Pi:
     ssh pi@rpi-west.local

   Create the service file:
     sudo nano /etc/systemd/system/adsb-ml-collector.service

   Paste the following (adjust paths and credentials as needed):

   ------- /etc/systemd/system/adsb-ml-collector.service -------
   [Unit]
   Description=ADS-B ML Training Data Collector
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=asaj
   WorkingDirectory=/home/asaj/workspace/adsb_autoencoder
   ExecStart=/home/asaj/workspace/adsb_autoencoder/.venv/bin/python \\
       ml_data_collector.py \\
       --antenna-id rpi_west \\
       --minio-endpoint rpi-master.local:9000 \\
       --minio-access-key adsbadmin \\
       --minio-secret-key adsbpass \\
       --minio-bucket adsb-iq-ml \\
       --output-dir /home/asaj/workspace/adsb_autoencoder/ml_data \\
       --burst-sec 30 \\
       --gains 28 32 36 40 44 \\
       --log-level INFO
   Restart=always
   RestartSec=10
   StandardOutput=journal
   StandardError=journal
   SyslogIdentifier=adsb-ml-collector

   [Install]
   WantedBy=multi-user.target
   ------- end of service file -------

   Enable and start:
     sudo systemctl daemon-reload
     sudo systemctl enable adsb-ml-collector.service
     sudo systemctl start  adsb-ml-collector.service

   Check status:
     sudo systemctl status adsb-ml-collector.service
     journalctl -u adsb-ml-collector.service -f   # live log tail

   Stop when needed:
     sudo systemctl stop adsb-ml-collector.service

6. REPEAT STEP 5 FOR rpi_east (change --antenna-id to rpi_east)

------------------------------------------------------------------------------
DOWNLOADING FOR TRAINING
------------------------------------------------------------------------------
From your Mac, once you have captured enough data:

  # Install mc if not already done
  brew install minio/stable/mc
  mc alias set rpi-minio http://rpi-master.local:9000 adsbadmin adsbpass

  # Download everything from the ML bucket
  mc cp --recursive rpi-minio/adsb-iq-ml ~/adsb_iq_data_ml/

  # Then run the extraction pipeline
  python extract_labels.py \\
      --data-dir  ~/adsb_iq_data_ml \\
      --labels-dir ~/adsb_iq_data_ml/labels \\
      --recursive

  # Fine-tune the model
  python train_supervised.py \\
      --labels-dir ~/adsb_iq_data_ml/labels \\
      --warm-start checkpoints/best_supervised.pt \\
      --base-channels 64 \\
      --ckpt-out checkpoints/best_supervised_v2.pt \\
      --epochs 50
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional MinIO import
# ---------------------------------------------------------------------------

try:
    from minio import Minio
    from minio.error import S3Error
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False

log = logging.getLogger("ml_collector")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_burst(
    antenna_id: str,
    device_index: int,
    gain_db: float,
    n_samples: int,
    output_dir: Path,
    rtl_sdr_bin: str = "rtl_sdr",
    timeout_sec: float = 45.0,
) -> Optional[Path]:
    """
    Capture a single IQ burst and save as a complex64 .npy file.

    Filename format:  <antenna_id>_<timestamp>_g<gain>.npy
    Companion JSON :  <antenna_id>_<timestamp>_g<gain>.json

    Parameters
    ----------
    gain_db   : SDR gain in dB.  Embedded in the filename for training-time filtering.
    n_samples : Total IQ samples to capture.

    Returns
    -------
    Path to the saved .npy file, or None on failure.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    gain_tag = f"g{int(gain_db):03d}"
    base_name = f"{antenna_id}_{ts}_{gain_tag}"

    npy_path  = output_dir / f"{base_name}.npy"
    json_path = output_dir / f"{base_name}.json"

    expected_bytes = n_samples * 2  # I byte + Q byte per sample

    # Pre-allocate memory-mapped numpy array
    iq_mm = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.complex64, shape=(n_samples,)
    )

    cmd = [
        rtl_sdr_bin,
        "-f", str(int(1_090_000_000)),
        "-s", str(int(2_000_000)),
        "-d", str(device_index),
        "-g", str(gain_db),
        "-n", str(n_samples),
        "-",
    ]

    log.debug("rtl_sdr cmd: %s", " ".join(cmd))

    bytes_read = 0
    sample_idx = 0

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        ) as proc:
            t0 = time.monotonic()

            while bytes_read < expected_bytes:
                if time.monotonic() - t0 > timeout_sec:
                    proc.kill()
                    raise TimeoutError(f"rtl_sdr timed out after {timeout_sec}s")

                chunk = proc.stdout.read(262_144)
                if not chunk:
                    break

                buf = np.frombuffer(chunk, dtype=np.uint8)
                if buf.size % 2 == 1:
                    buf = buf[:-1]

                n_iq = buf.size // 2
                I = (buf[0::2].astype(np.float32) - 127.5) / 127.5
                Q = (buf[1::2].astype(np.float32) - 127.5) / 127.5

                end_idx = sample_idx + n_iq
                if end_idx > n_samples:
                    n_iq   = n_samples - sample_idx
                    I, Q   = I[:n_iq], Q[:n_iq]
                    end_idx = n_samples

                # Avoid the complex128 intermediate that (I + 1j*Q) would create.
                # Write real and imaginary parts separately into a pre-typed view.
                chunk_iq = np.empty(n_iq, dtype=np.complex64)
                chunk_iq.real[:] = I
                chunk_iq.imag[:] = Q
                iq_mm[sample_idx:end_idx] = chunk_iq
                del chunk_iq, I, Q, buf
                sample_idx = end_idx
                bytes_read += buf.size

                if sample_idx >= n_samples:
                    break

            proc.wait(timeout=5)
            stderr_out = proc.stderr.read().decode(errors="ignore").strip()
            if proc.returncode not in (0, None) and proc.returncode != 0:
                raise RuntimeError(
                    f"rtl_sdr exit {proc.returncode}: {stderr_out[:200]}"
                )

        if sample_idx < n_samples:
            raise RuntimeError(
                f"Only captured {sample_idx}/{n_samples} samples"
            )

        del iq_mm  # flush memmap

        # Save metadata sidecar
        meta = {
            "antenna_id":         antenna_id,
            "device_index":       device_index,
            "center_frequency_hz": 1_090_000_000,
            "sample_rate_sps":    2_000_000,
            "gain_db":            gain_db,
            "num_samples":        n_samples,
            "burst_duration_sec": n_samples / 2_000_000,
            "start_time_utc":     datetime.now(timezone.utc).isoformat(),
            "filename_iq":        npy_path.name,
            "collector":          "ml_data_collector",
        }
        json_path.write_text(json.dumps(meta, indent=2))

        log.info(
            "Captured: %s  gain=%.0f dB  samples=%d  size=%.1f MB",
            npy_path.name, gain_db, n_samples, npy_path.stat().st_size / 1e6
        )
        return npy_path

    except Exception as exc:
        log.error("Capture failed (gain=%.0f dB): %s", gain_db, exc)
        for p in (npy_path, json_path):
            if p.exists():
                p.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------------
# MinIO upload
# ---------------------------------------------------------------------------

def make_minio_client(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool = False,
    bucket: str = "adsb-iq-ml",
) -> Optional["Minio"]:
    """Create and validate a MinIO client, ensuring the bucket exists."""
    if not HAS_MINIO:
        log.warning("minio package not installed — uploads disabled.  pip install minio")
        return None
    try:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            log.info("Created MinIO bucket: %s", bucket)
        log.info("MinIO connected: %s  bucket=%s", endpoint, bucket)
        return client
    except Exception as exc:
        log.warning("MinIO init failed (%s) — will save locally only.", exc)
        return None


def upload_file(
    client: "Minio",
    bucket: str,
    local_path: Path,
    antenna_id: str,
) -> bool:
    """Upload a file to MinIO under antenna_id/YYYY/MM/DD/<filename>."""
    today = datetime.now(timezone.utc)
    key = (
        f"{antenna_id}/{today.year:04d}/{today.month:02d}/{today.day:02d}"
        f"/{local_path.name}"
    )
    try:
        client.fput_object(bucket, key, str(local_path))
        log.debug("Uploaded %s → %s/%s", local_path.name, bucket, key)
        return True
    except Exception as exc:
        log.warning("Upload failed for %s: %s", local_path.name, exc)
        return False


# ---------------------------------------------------------------------------
# Main collector loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(2_000_000 * args.burst_sec)
    gains: List[float] = args.gains
    gain_idx = 0

    log.info(
        "ML collector started — antenna=%s  burst=%.0fs  gains=%s  output=%s",
        args.antenna_id, args.burst_sec, gains, output_dir,
    )

    # Connect to MinIO (optional — falls back to local if unavailable)
    minio_client = None
    if args.minio_endpoint:
        minio_client = make_minio_client(
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            secure=args.minio_secure,
            bucket=args.minio_bucket,
        )

    # Graceful shutdown on SIGINT / SIGTERM
    stop = {"flag": False}
    def _handle_signal(signum, frame):
        log.info("Signal %s received — stopping after current burst.", signum)
        stop["flag"] = True
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    burst_count = 0
    upload_count = 0
    fail_count = 0

    while not stop["flag"]:
        gain = gains[gain_idx % len(gains)]
        gain_idx += 1

        npy_path = capture_burst(
            antenna_id=args.antenna_id,
            device_index=args.device_index,
            gain_db=gain,
            n_samples=n_samples,
            output_dir=output_dir,
            rtl_sdr_bin=args.rtl_sdr_bin,
            timeout_sec=args.burst_sec + 15.0,
        )

        if npy_path is None:
            fail_count += 1
            if fail_count >= args.max_failures:
                log.error(
                    "Reached %d consecutive failures — stopping.", args.max_failures
                )
                break
            time.sleep(2.0)
            continue

        fail_count = 0
        burst_count += 1

        # Upload both .npy and .json sidecar
        if minio_client:
            for path in (npy_path, npy_path.with_suffix(".json")):
                if path.exists():
                    ok = upload_file(minio_client, args.minio_bucket, path, args.antenna_id)
                    if ok:
                        upload_count += 1
                        if not args.keep_local:
                            path.unlink(missing_ok=True)
                    else:
                        log.warning(
                            "Upload failed — keeping %s locally.", path.name
                        )

        log.info(
            "Progress: bursts=%d  uploaded=%d  gain_cycle=%d/%d",
            burst_count, upload_count, (gain_idx % len(gains)) + 1, len(gains),
        )

        # Periodic forced GC to prevent slow memory accumulation over long runs.
        gc.collect()

        # Self-restart: exit cleanly after --max-bursts so systemd (Restart=always)
        # relaunches the process with a fresh heap.  This is the recommended pattern
        # for long-running collection services on memory-constrained hardware.
        if args.max_bursts and burst_count >= args.max_bursts:
            log.info(
                "Reached --max-bursts=%d — exiting for clean systemd restart.",
                args.max_bursts,
            )
            break

    log.info(
        "Collector stopped.  Total bursts: %d  uploaded: %d",
        burst_count, upload_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ML-optimised ADS-B IQ collector with gain rotation and MinIO upload.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sdr = p.add_argument_group("SDR")
    sdr.add_argument("--antenna-id",    default="rpi_west",
                     help="Identifier for this antenna/device (used in filenames).")
    sdr.add_argument("--device-index",  type=int, default=0,
                     help="RTL-SDR device index (0 if only one dongle).")
    sdr.add_argument("--burst-sec",     type=float, default=30.0,
                     help="Duration of each IQ burst in seconds.  "
                          "30s → ~60M samples → ~480 MB per file.")
    sdr.add_argument("--gains",         type=float, nargs="+",
                     default=[28.0, 32.0, 36.0, 40.0, 44.0],
                     help="Gain values (dB) to cycle through.  Each consecutive "
                          "burst uses the next value in the list.  More values = "
                          "more SNR diversity in training data.")
    sdr.add_argument("--rtl-sdr-bin",   default="rtl_sdr",
                     help="Path to the rtl_sdr binary.")

    stor = p.add_argument_group("Storage")
    stor.add_argument("--output-dir",   default="ml_data",
                      help="Local directory for .npy files.  Used as a staging "
                           "area before MinIO upload (or permanent if no MinIO).")
    stor.add_argument("--keep-local",   action="store_true",
                      help="Keep local .npy files even after successful upload.  "
                           "By default, uploaded files are deleted locally to save space.")

    minio = p.add_argument_group("MinIO (central Pi)")
    minio.add_argument("--minio-endpoint",   default="rpi-master.local:9000",
                       help="MinIO server host:port.  Port 9000 = S3 API.")
    minio.add_argument("--minio-access-key", default="adsbadmin")
    minio.add_argument("--minio-secret-key", default="adsbpass")
    minio.add_argument("--minio-bucket",     default="adsb-iq-ml",
                       help="Separate bucket from the general collector to avoid mixing.")
    minio.add_argument("--minio-secure",     action="store_true",
                       help="Use HTTPS for MinIO (only if TLS is configured).")
    minio.add_argument("--no-minio",         action="store_true",
                       help="Disable MinIO upload entirely — save locally only.")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--max-failures", type=int, default=5,
                      help="Stop after this many consecutive capture failures.")
    misc.add_argument("--max-bursts",   type=int, default=60,
                      help="Exit cleanly after this many bursts so that systemd "
                           "(Restart=always) relaunches the process with a fresh "
                           "heap.  Prevents slow memory accumulation on constrained "
                           "hardware (RPi 3B/4).  Default 60 bursts × 30 s = 30 min "
                           "per session.  Set to 0 to disable.")
    misc.add_argument("--log-level",    default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.no_minio:
        args.minio_endpoint = None

    run(args)


if __name__ == "__main__":
    main()
