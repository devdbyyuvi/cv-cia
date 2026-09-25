# Feed-Forward Inverse Rendering from Sparse Casual Captures

A PyTorch research pipeline that reconstructs **geometry (SDF/normals/depth)**,
**spatially varying BRDF (albedo, roughness, metallic)**, and **environment
illumination (spherical Gaussians)** from a handful (3–6) of casual multi-view
photos, using a feed-forward Hexa-Plane backbone regularized by a pretrained
material diffusion prior, with explicit uncertainty quantification and a
strict held-out-illumination relighting benchmark.

This repository is organized around the 5-phase project plan:

| Phase | Focus | Key modules |
|---|---|---|
| 1 | Environment setup & data pipeline | `src/data`, `src/rendering/differentiable_renderer.py` |
| 2 | Feed-forward architecture & geometry backbone | `src/models/hexaplane.py`, `src/models/heads.py`, `src/training/train_stage1.py` |
| 3 | Diffusion-prior regularization & uncertainty | `src/diffusion_prior`, `src/losses/ambiguity_loss.py`, `src/uncertainty` |
| 4 | Fine-tuning & relighting validation | `src/training/finetune_real.py`, `src/evaluation/relighting_eval.py` |
| 5 | Documentation, reporting, demo | `docs/`, `src/visualization`, `demo/run_demo.py` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1: sanity-check data loading + differentiable renderer
python -m tests.test_rendering

# Phase 2: train the feed-forward backbone on synthetic data
python -m src.training.train_stage1 --config configs/default.yaml

# Phase 3: continue training with diffusion-prior regularization + uncertainty
python -m src.training.train_stage2_diffusion --config configs/default.yaml \
    --init_ckpt runs/stage1/last.ckpt

# Phase 4: fine-tune on real casual captures, then run relighting validation
python -m src.training.finetune_real --config configs/default.yaml \
    --init_ckpt runs/stage2/last.ckpt --capture_dir data/real_captures/object_01
python -m src.evaluation.relighting_eval --ckpt runs/finetune/last.ckpt \
    --capture_dir data/real_captures/object_01 --novel_env data/envmaps/held_out.hdr

# Phase 4: benchmark inference speed
python -m src.evaluation.benchmark_inference --ckpt runs/finetune/last.ckpt

# Phase 5: run the end-to-end demo (photos -> editable 3D asset -> relit renders)
python demo/run_demo.py --images data/real_captures/object_01 --out demo/output
```

## Repository layout

```
configs/                 YAML experiment configs
src/data/                Synthetic + real capture dataset loaders
src/models/              Hexa-plane backbone, SDF decoder, prediction heads, SG lighting
src/rendering/           Differentiable renderer, BRDF (Cook-Torrance/GGX), SG relighting
src/losses/              Photometric, mask, ambiguity/physical-plausibility, diffusion-prior losses
src/diffusion_prior/     Wrapper around a pretrained material diffusion prior (MaterialFusion / IntrinsicAnything style)
src/uncertainty/         Monte-Carlo / latent-sampling disagreement -> uncertainty maps
src/training/            Stage 1 (feed-forward), Stage 2 (diffusion-regularized), real fine-tuning loops
src/evaluation/          Relighting validation suite, inference benchmarking, ablation harness
src/visualization/       Open3D desktop viewer + lightweight web viewer
src/utils/               Camera, I/O, logging utilities
scripts/                 Shell helpers (data download, full pipeline runner)
tests/                   Lightweight unit/smoke tests
docs/                    Architecture diagram (Mermaid), phase plan, report template
demo/                    End-to-end demo script producing editable assets + relit renders
```

## Notes on external dependencies

- **Diffusion prior weights** (`src/diffusion_prior/material_prior.py`): this repo
  wraps a pretrained material diffusion prior (e.g. a MaterialFusion- or
  IntrinsicAnything-style checkpoint) via a thin adapter interface. It does not
  vendor third-party weights; point `configs/default.yaml -> diffusion_prior.ckpt_path`
  at a locally downloaded checkpoint, or set `diffusion_prior.enabled: false`
  to fall back to an optimization-only (no-prior) baseline for ablations.
- **Differentiable rendering**: `src/rendering/differentiable_renderer.py` implements a
  self-contained analytic Cook-Torrance/GGX rasterization-based renderer so the
  project runs without a heavyweight external DR dependency. Swapping in
  `nvdiffrast` or `redner` is a drop-in change (see comments in that file).
- **3–6 photo real captures**: see `data/real_captures/README.md` for the expected
  folder format (images + COLMAP poses) and `scripts/download_data.sh` for
  fetching example synthetic datasets.

See `docs/architecture.md` for the full architecture diagram and `docs/phase_plan.md`
for the detailed phase-by-phase engineering plan (mirrors the project brief).
