"""
inference/scripts/sync_to_pod.sh

Sync code moi len RunPod pod qua SCP.

Su dung:
  bash inference/scripts/sync_to_pod.sh

Can setting bien moi truong:
  POD_HOST   - SSH host cua pod (e.g. "69.30.85.25")
  POD_PORT   - SSH port cua pod (e.g. 22057)
  POD_USER   - SSH user (default: root)
  POD_KEY    - SSH key path (default: $env:USERPROFILE/.ssh/id_ed25519)
  POD_DIR    - Thu muc tren pod (default: /workspace/isometric-map)

Hoac pass args:
  bash inference/scripts/sync_to_pod.sh user@host:22057 /workspace/isometric-map
"""
#!/usr/bin/env bash
set -e

POD_HOST="${POD_HOST:-}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_DIR="${POD_DIR:-/workspace/isometric-map}"

if [ -n "$1" ]; then
    # Parse user@host:port
    ARG1="$1"
    if [[ "$ARG1" == *":"* ]]; then
        POD_USER="${ARG1%%@*}"
        REMAIN="${ARG1#*@}"
        POD_HOST="${REMAIN%%:*}"
        POD_PORT="${REMAIN#*:}"
    else
        POD_HOST="$ARG1"
    fi
fi

if [ -n "$2" ]; then
    POD_DIR="$2"
fi

if [ -z "$POD_HOST" ]; then
    echo "ERROR: POD_HOST not set. Either export POD_HOST=ip or pass as 1st arg."
    echo "Usage: $0 [user@]host[:port] [/remote/path]"
    exit 1
fi

SSH_TARGET="$POD_USER@$POD_HOST"
SSH_OPTS=(-i "$POD_KEY" -p "$POD_PORT" -o StrictHostKeyChecking=no)
SSH="${SSH} ${SSH_OPTS[@]}"
SCP="scp ${SSH_OPTS[@]}"

echo "Pod SSH:    $SSH_TARGET port $POD_PORT"
echo "Pod dir:    $POD_DIR"
echo "Pod key:    $POD_KEY"
echo

# Files can sync (chi gui file da sua)
FILES=(
    "inference/config.py"
    "inference/__init__.py"
    "inference/client/__init__.py"
    "inference/client/template.py"
    "inference/client/traversal.py"
    "inference/scripts/run_inference_pipeline.py"
    "inference/scripts/test_template_logic.py"
    "inference/scripts/test_quadrant_local.py"
)

# Check files exist
MISSING=0
for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: local file not found: $f"
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    exit 1
fi

# Make sure pod dir exists
echo "[1/3] mkdir -p $POD_DIR on pod..."
$SSH $SSH_TARGET "mkdir -p $POD_DIR"

# Sync each file
echo
echo "[2/3] Syncing files..."
for f in "${FILES[@]}"; do
    echo "  -> $f"
    $SCP "$f" "$SSH_TARGET:$POD_DIR/$f"
done

# Restart server
echo
echo "[3/3] Restart server (if start_server.sh exists)..."
$SSH $SSH_TARGET "cd $POD_DIR && (test -f start_server.sh && (pkill -f 'uvicorn inference.server.main' || true) && sleep 2 && nohup ./start_server.sh > server.log 2>&1 &) || echo 'No start_server.sh, please restart manually'"

echo
echo "Done. Server should be ready in ~30-60s."
echo
echo "Next step: from LOCAL machine, run:"
echo "  ssh -i $POD_KEY -p $POD_PORT -L 10100:127.0.0.1:10100 -N $SSH_TARGET"
echo "  py -3.11 -m inference.scripts.run_inference_pipeline --renders output/renders --output model_generate_v2 --endpoint http://127.0.0.1:10100 --concurrency 1"
