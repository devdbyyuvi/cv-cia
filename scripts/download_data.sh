#!/usr/bin/env bash
set -euo pipefail

DEST="data/synthetic"
mkdir -p "$DEST"

if [ -n "${DATASET_URL:-}" ]; then
    echo "Downloading $DATASET_URL -> $DEST ..."
    curl -sSL "$DATASET_URL" -o /tmp/synthetic_dataset.tar.gz
    tar -xzf /tmp/synthetic_dataset.tar.gz -C "$DEST"
    rm -f /tmp/synthetic_dataset.tar.gz
    echo "Done. Scenes extracted to $DEST"
else
    echo "No DATASET_URL provided."
    echo "Data loaders will use local scenes in $DEST or fall back to dummy samples."
fi