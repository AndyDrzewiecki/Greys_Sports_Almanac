#!/usr/bin/env bash
# ============================================================
# EliteBook 840 G2 (Ubuntu) — Ollama inference server setup
# for Market Watch Agents (secondary/extraction model host)
# Run as: bash elitebook_ollama.sh  (no sudo needed for Ollama)
# ============================================================
set -euo pipefail

EXTRACTION_MODEL="${EXTRACTION_MODEL:-llama3.1:8b}"
FAST_MODEL="${FAST_MODEL:-llama3.1:8b}"

echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Enabling Ollama to listen on LAN (not just localhost) ==="
# Create a systemd override so Ollama binds to all interfaces
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

systemctl daemon-reload
systemctl enable ollama
systemctl start ollama

echo "=== Waiting for Ollama to be ready ==="
sleep 5

echo "=== Pulling models ==="
ollama pull "${EXTRACTION_MODEL}"
ollama pull "${FAST_MODEL}"

echo ""
echo "=== DONE ==="
ELITEBOOK_IP=$(hostname -I | awk '{print $1}')
echo "EliteBook LAN IP: ${ELITEBOOK_IP}"
echo ""
echo "Add these to mwa/.env on the ZBook:"
echo ""
echo "OLLAMA_SECONDARY_URL=http://${ELITEBOOK_IP}:11434"
echo "OLLAMA_EXTRACTION_MODEL=${EXTRACTION_MODEL}"
echo "OLLAMA_FAST_MODEL=${FAST_MODEL}"
echo ""
echo "Test the connection from the ZBook:"
echo "  curl http://${ELITEBOOK_IP}:11434/api/tags"
