#!/usr/bin/env bash
# deploy.sh — Sync adsb_autoencoder from Mac to Raspberry Pi
# ===========================================================
#
# Usage
# -----
#   ./deploy.sh                          # dry-run preview (no changes)
#   ./deploy.sh --push                   # actually sync files
#   ./deploy.sh --push --install-deps    # sync + pip install on Pi
#
# Environment overrides (set before running or export in your shell):
#   RPI_USER   Pi username        (default: pi)
#   RPI_HOST   Pi hostname or IP  (default: raspberrypi.local)
#   RPI_DIR    Remote target dir  (default: /home/$RPI_USER/adsb_autoencoder)
#
# Example:
#   RPI_HOST=192.168.1.42 ./deploy.sh --push --install-deps

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
RPI_USER="${RPI_USER:-asaj}"
RPI_HOST="${RPI_HOST:-rpi-west.local}"
RPI_DIR="${RPI_DIR:-/home/${RPI_USER}/workspace/adsb_autoencoder}"

# Resolve the directory that contains this script (handles symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
DO_PUSH=0
INSTALL_DEPS=0

for arg in "$@"; do
  case "$arg" in
    --push)          DO_PUSH=1 ;;
    --install-deps)  INSTALL_DEPS=1 ;;
    --help|-h)
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

# ── rsync options ─────────────────────────────────────────────────────────────
#
# INCLUSION STRATEGY — whitelist code files only.
# rsync filter rules are evaluated top-to-bottom; the first match wins.
# We explicitly include every file type that belongs on the Pi, then exclude
# everything else.  This means large data files (.npy, .npz, .bin, ml_data/,
# labels/, etc.) are never transferred even if they are present locally.
#
RSYNC_OPTS=(
  --archive          # preserve permissions, timestamps, symlinks, etc.
  --compress         # reduce bandwidth over WiFi/SSH
  --human-readable   # human-readable transfer sizes
  --progress         # per-file progress bar
  --delete           # remove remote files that no longer exist locally

  # ── Explicit inclusions (code + config only) ─────────────────────────────
  --include="*.py"              # all Python source files
  --include="*.sh"              # shell scripts (deploy.sh itself, etc.)
  --include="*.txt"             # requirements_rpi.txt, any plain-text configs
  --include="*.md"              # README / blueprint docs
  --include="docs/"             # technical report (PAPER.md, paper.pdf, …)
  --include="docs/**"
  --include="data/README.md"    # data layout notes
  --include="artifacts/README.md"
  --include="*.json"            # small config/metadata JSON files

  # ── Exclusions — data, artefacts, and environment dirs ───────────────────
  --exclude="*.npy"             # raw IQ burst files (can be 100s of GB)
  --exclude="*.npz"             # label files produced by extract_labels.py
  --exclude="*.bin"             # RTL-SDR binary captures
  --exclude="*.pt"              # PyTorch checkpoints (too large; copy manually)
  --exclude="*.pth"
  --exclude="*.png"             # evaluation / visualisation plots
  --exclude="*.jpg"
  --exclude="*.jpeg"
  --exclude="ml_data/"          # local IQ staging directory
  --exclude="labels/"           # extracted label files
  --exclude="checkpoints/"      # trained weights
  --exclude="evaluation_results*"
  --exclude=".venv/"
  --exclude=".venv_*/"
  --exclude="__pycache__/"
  --exclude="*.pyc"
  --exclude="*.pyo"
  --exclude=".DS_Store"
  --exclude=".git/"
  --exclude=".cursor/"

  # Catch-all: exclude anything not matched by the inclusions above
  --exclude="*"
)

REMOTE="${RPI_USER}@${RPI_HOST}:${RPI_DIR}/"
SOURCE="${SCRIPT_DIR}/"

# ── Dry-run preview (default — no changes) ───────────────────────────────────
echo "============================================================"
echo " ADS-B Autoencoder — Deployment to Raspberry Pi"
echo "============================================================"
echo " Source : ${SOURCE}"
echo " Target : ${REMOTE}"
echo " Mode   : $([ "$DO_PUSH" -eq 1 ] && echo 'LIVE SYNC' || echo 'DRY RUN (pass --push to sync)')"
echo "============================================================"

if [ "$DO_PUSH" -eq 0 ]; then
  # Show what WOULD be transferred without actually doing it
  rsync "${RSYNC_OPTS[@]}" --dry-run --itemize-changes \
    "${SOURCE}" "${REMOTE}"
  echo ""
  echo "  ↑ Dry-run complete.  Run with --push to apply."
  exit 0
fi

# ── Live sync ─────────────────────────────────────────────────────────────────
echo "[deploy] Syncing files …"
rsync "${RSYNC_OPTS[@]}" "${SOURCE}" "${REMOTE}"
echo "[deploy] Sync complete."

# ── Optional: remote dependency install ──────────────────────────────────────
if [ "$INSTALL_DEPS" -eq 1 ]; then
  echo "[deploy] Setting up venv + installing requirements_rpi.txt on ${RPI_HOST} …"
  ssh "${RPI_USER}@${RPI_HOST}" bash -s <<REMOTE_SCRIPT
    set -euo pipefail
    cd "${RPI_DIR}"

    # Create the venv only if it doesn't exist yet (idempotent)
    if [ ! -d .venv ]; then
      echo "[remote] Creating virtual environment …"
      python3 -m venv .venv
    else
      echo "[remote] Virtual environment already exists — reusing."
    fi

    source .venv/bin/activate
    echo "[remote] Python: \$(python3 --version)"
    echo "[remote] pip:    \$(pip --version)"
    pip install --upgrade pip --quiet
    # requirements_rpi.txt points pip at the PyTorch CPU-only wheel index
    # which carries explicit +cpu aarch64 builds — no CUDA libs are pulled in.
    pip install -r requirements_rpi.txt
    echo "[remote] Disk usage: \$(du -sh .venv/lib 2>/dev/null | cut -f1)"

    echo "[remote] Installation complete."
    python3 -c "import torch, numpy; print('[remote] torch', torch.__version__, '| numpy', numpy.__version__, '| CUDA available:', torch.cuda.is_available())"
REMOTE_SCRIPT
  echo "[deploy] Remote installation done."
fi

echo ""
echo "============================================================"
echo " Deployment complete!"
echo " Connect to the Pi and run:"
echo "   cd ${RPI_DIR}"
echo "   source .venv/bin/activate"
echo "   python3 live_bridge.py --ckpt checkpoints/best.pt \\"
echo "       --torchscript --threads 4 --hop 240 --batch-size 16"
echo "============================================================"
