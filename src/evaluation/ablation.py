"""Phase 5: ablation harness comparing the diffusion-prior-regularized model
against an optimization-only baseline (diffusion_prior.enabled=false), on the
same held-out relighting evaluation set, to quantify the prior's contribution
to resolving material-lighting ambiguity.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List

from src.evaluation.relighting_eval import run_relighting_eval
from src.utils.io_utils import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_ablation(
    base_config_path: str,
    checkpoints: Dict[str, str],
    eval_cases: List[Dict[str, str]],
    out_path: str = "runs/ablation_results.json",
):
    """
    checkpoints: {"with_prior": "runs/stage2/last.ckpt", "no_prior": "runs/stage2_no_prior/last.ckpt"}
    eval_cases: list of dicts with keys capture_dir, novel_env, gt_relit_photo, held_out_pose
    """
    results = {}
    for variant, ckpt_path in checkpoints.items():
        variant_results = []
        for case in eval_cases:
            metrics = run_relighting_eval(
                ckpt_path=ckpt_path,
                capture_dir=case["capture_dir"],
                novel_env_sg_path=case["novel_env"],
                gt_relit_photo_path=case["gt_relit_photo"],
                held_out_pose_path=case["held_out_pose"],
                cfg_path=base_config_path,
            )
            variant_results.append({"object": case["capture_dir"], **metrics})
        avg = {
            k: sum(r[k] for r in variant_results) / len(variant_results)
            for k in ("psnr", "ssim", "lpips")
        }
        results[variant] = {"per_object": variant_results, "average": avg}
        logger.info(f"[ablation:{variant}] average -> PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} LPIPS={avg['lpips']:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation results written to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--with_prior_ckpt", type=str, required=True)
    parser.add_argument("--no_prior_ckpt", type=str, required=True)
    parser.add_argument("--eval_manifest", type=str, required=True,
                         help="JSON list of {capture_dir, novel_env, gt_relit_photo, held_out_pose}")
    parser.add_argument("--out", type=str, default="runs/ablation_results.json")
    args = parser.parse_args()

    with open(args.eval_manifest) as f:
        eval_cases = json.load(f)

    run_ablation(
        args.config,
        {"with_diffusion_prior": args.with_prior_ckpt, "optimization_only": args.no_prior_ckpt},
        eval_cases,
        args.out,
    )


if __name__ == "__main__":
    main()
