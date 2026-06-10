#!/usr/bin/env bash
# Deploy face-guard to a Vast.ai instance.
# Usage: bash deploy.sh <SSH_PORT> <HOST>
# Example: bash deploy.sh 47331 108.55.118.247

set -euo pipefail

SSH_PORT="${1:?Usage: deploy.sh <SSH_PORT> <HOST>}"
HOST="${2:?Usage: deploy.sh <SSH_PORT> <HOST>}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="root@$HOST"
SSH_OPTS="-i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p $SSH_PORT"

echo "==> Packaging project (Python files + models, excluding __pycache__ and tests)"
tar --exclude='**/__pycache__' \
    --exclude='**/test*' \
    --exclude='**/*.pyc' \
    --exclude='.git' \
    -czf /tmp/face-guard.tar.gz \
    -C "$(dirname "$0")" .

echo "==> Uploading to $REMOTE"
scp $SSH_OPTS /tmp/face-guard.tar.gz "$REMOTE:/workspace/face-guard.tar.gz"

echo "==> Extracting and installing on remote"
ssh $SSH_OPTS "$REMOTE" bash <<'REMOTE_SCRIPT'
set -euo pipefail
mkdir -p /workspace/face-guard
tar -xzf /workspace/face-guard.tar.gz -C /workspace/face-guard
cd /workspace/face-guard

echo "-- Installing system deps"
apt-get update -q && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 > /dev/null

echo "-- Installing Python deps"
pip install --quiet --no-cache-dir -r requirements.txt

echo "-- Checking .env"
if [ ! -f .env ]; then
  echo "ERROR: .env file not found. Copy .env.example to .env and fill in values."
  exit 1
fi

echo "-- Starting API (uvicorn, background)"
pkill -f "uvicorn app.server" 2>/dev/null || true
nohup uvicorn app.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  > /workspace/api.log 2>&1 &
echo "API started. Logs: /workspace/api.log"
sleep 3
curl -sf http://localhost:8000/api/v1/health | python3 -m json.tool || echo "Health check failed — check /workspace/api.log"
REMOTE_SCRIPT

echo "==> Done. API is running on $HOST:8000"
rm -f /tmp/face-guard.tar.gz
