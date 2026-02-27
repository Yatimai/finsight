#!/usr/bin/env bash
# setup_runpod.sh — Bootstrap a RunPod H200 pod for finsight indexing.
# Idempotent: safe to run multiple times.
set -euo pipefail

REPO_URL="${FINSIGHT_REPO_URL:-https://github.com/YOUR_USER/finsight.git}"
REPO_DIR="${FINSIGHT_DIR:-/workspace/finsight}"
QDRANT_VOLUME="/workspace/qdrant_storage"

echo "=== [1/6] System packages ==="
apt-get update -qq
apt-get install -y -qq poppler-utils git curl > /dev/null

echo "=== [2/6] Clone / pull repo ==="
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== [3/6] Python dependencies ==="
# PyTorch with CUDA 12.4 (RunPod H200 default)
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

pip install -q -r requirements.txt

# flash-attn (optional, speeds up ColQwen2 ~2x)
echo "Installing flash-attn (may take a few minutes, non-fatal if it fails)..."
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "WARN: flash-attn failed to install, will use default attention"

echo "=== [4/6] Qdrant (Docker) ==="
if docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
    echo "Qdrant container already running."
else
    docker rm -f qdrant 2>/dev/null || true
    mkdir -p "$QDRANT_VOLUME"
    docker run -d \
        --name qdrant \
        --restart unless-stopped \
        -p 6333:6333 \
        -p 6334:6334 \
        -v "$QDRANT_VOLUME:/qdrant/storage" \
        qdrant/qdrant:latest
    echo "Qdrant started on localhost:6333"
fi

# Wait for Qdrant to be ready
echo "Waiting for Qdrant..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        echo "Qdrant is ready."
        break
    fi
    sleep 1
done

echo "=== [5/6] Config ==="
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    # Switch to remote mode for Docker Qdrant
    sed -i 's/mode: "embedded"/mode: "remote"/' config.yaml
    sed -i 's|# remote_url: "https://..."|remote_url: "http://localhost:6333"|' config.yaml
    echo "Created config.yaml (remote mode, localhost:6333)"
else
    echo "config.yaml already exists, skipping."
fi

echo "=== [6/6] Ready ==="
echo ""
echo "Setup complete. Next steps:"
echo ""
echo "  1. Copy PDFs into $REPO_DIR/data/documents/"
echo "  2. Dry-run (validate pipeline without writing):"
echo "     python -m indexing.index_documents --dry-run"
echo "  3. Index one document (e.g. Walmart 10K):"
echo "     python -m indexing.index_documents --pdf data/documents/Walmart_10K_2024.pdf"
echo "  4. Index all documents:"
echo "     python -m indexing.index_documents"
echo ""
