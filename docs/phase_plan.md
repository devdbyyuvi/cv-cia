# Phase Plan

This mirrors the 5-phase project brief and maps each bullet to the concrete
module(s) implementing it.

## Phase 1 -- Environment Setup & Baseline Data Pipeline
- **Repository & framework**: `setup.py`, `requirements.txt`, modular `src/{data,models,rendering,losses}` packages.
- **Synthetic dataset ingestion**: `src/data/synthetic_dataset.py` (GT SDF/depth, SVBRDF, envmap).
- **Casual smartphone capture curation**: `src/data/real_capture_dataset.py` (3-6 orbital photos + COLMAP poses); see `data/real_captures/README.md`.
- **Differentiable rendering**: `src/rendering/differentiable_renderer.py`, `src/rendering/brdf.py`, `src/rendering/sh_lighting.py`.

## Phase 2 -- Feed-Forward Architecture & Geometry Backbone
- **Hexa-Plane backbone**: `src/models/hexaplane.py`.
- **Decoupled heads**: `src/models/heads.py` (`GeometryHead`, `MaterialHead`, `IlluminationHead`), assembled in `src/models/network.py`.
- **Initial training loop**: `src/training/train_stage1.py` (photometric L1/L2 + mask + eikonal).

## Phase 3 -- Diffusion-Prior Regularization & Uncertainty Estimation
- **Material diffusion prior integration**: `src/diffusion_prior/material_prior.py` (adapter interface + integration point for MaterialFusion/IntrinsicAnything-style checkpoints).
- **Ambiguity penalty**: `src/losses/ambiguity_loss.py` (energy conservation + prior-consistency terms), `src/losses/diffusion_prior_loss.py` (SDS-style distillation).
- **Uncertainty quantification**: `src/uncertainty/uncertainty_estimation.py` (MC-dropout / latent sampling disagreement -> uncertainty map).
- **Training loop**: `src/training/train_stage2_diffusion.py`.

## Phase 4 -- Fine-Tuning & Relighting Validation
- **Real-world fine-tuning**: `src/training/finetune_real.py` (self-supervised photometric consistency; material head lightly anchored to the Stage-2 solution).
- **Relighting validation suite**: `src/evaluation/relighting_eval.py` (render under a held-out novel environment, compare to a real photo via PSNR/SSIM/LPIPS).
- **Performance benchmarking**: `src/evaluation/benchmark_inference.py`.

## Phase 5 -- Documentation, Reporting & Final Demo
- **Ablation studies**: `src/evaluation/ablation.py` (diffusion-prior vs. optimization-only baseline).
- **Visualization/UI**: `src/visualization/viewer.py` (Open3D desktop viewer, marching-cubes mesh extraction), `src/visualization/web_viewer/index.html` (lightweight three.js viewer).
- **Final report & demo**: `docs/architecture.md`, this file, and `demo/run_demo.py` (end-to-end photos -> editable 3D asset -> relit renders).

## Suggested milestones / timeline

| Week | Milestone |
|---|---|
| 1 | Phase 1 complete: data loaders run on a small synthetic subset; renderer unit-tested (`tests/test_rendering.py`) |
| 2-3 | Phase 2: Stage-1 model trains and overfits a handful of synthetic scenes; sanity-check reconstructions |
| 4 | Phase 2 continued: full synthetic-dataset training run, tune losses |
| 5 | Phase 3: integrate a real (or stubbed) material diffusion prior; verify the ambiguity penalty measurably changes material/lighting split on a synthetic ambiguity stress-test scene |
| 6 | Phase 3 continued: uncertainty maps validated qualitatively against known-occluded regions |
| 7 | Phase 4: capture 3-5 real objects (3-6 photos each + COLMAP poses); fine-tune |
| 8 | Phase 4 continued: capture held-out-illumination ground truth photos; run relighting validation; benchmark inference speed |
| 9 | Phase 5: ablations, viewer polish, report + demo recording |
