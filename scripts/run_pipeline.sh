#!/usr/bin/env bash
# Runs the full pipeline end-to-end: Stage 1 -> Stage 2 (diffusion prior) ->
# real-capture fine-tuning -> relighting validation -> inference benchmark.
#
# Usage: bash scripts/run_pipeline.sh <capture_dir> [config]
set -euo pipefail

CAPTURE_DIR="${1:?Usage: run_pipeline.sh <capture_dir> [config]}"
CONFIG="${2:-configs/default.yaml}"

echo "== Phase 2: Stage 1 feed-forward training =="
python -m src.training.train_stage1 --config "$CONFIG" --out_dir runs/stage1

echo "== Phase 3: Stage 2 diffusion-prior-regularized training =="
python -m src.training.train_stage2_diffusion --config "$CONFIG" \
    --init_ckpt runs/stage1/last.ckpt --out_dir runs/stage2

echo "== Phase 4: fine-tuning on real capture: $CAPTURE_DIR =="
python -m src.training.finetune_real --config "$CONFIG" \
    --init_ckpt runs/stage2/last.ckpt --capture_dir "$CAPTURE_DIR" --out_dir runs/finetune

echo "== Phase 4: inference speed benchmark =="
python -m src.evaluation.benchmark_inference --ckpt runs/finetune/last.ckpt --config "$CONFIG"

echo "Pipeline complete. Run src/evaluation/relighting_eval.py separately once"
echo "held-out-illumination ground truth photos are available for $CAPTURE_DIR."
