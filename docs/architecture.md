# Architecture

## End-to-end data flow

```mermaid
flowchart TD
    A[3-6 sparse input photos] --> B[Image Encoder<br/>pooled multi-view feature]
    B --> C[Hexa-Plane Field<br/>6 learned feature planes]
    C --> D[SDF Decoder<br/>geometric init]
    C --> E[Geometry Head<br/>normals / depth]
    C --> F[Material Head<br/>albedo / roughness / metallic]
    B --> G[Illumination Head<br/>Spherical Gaussian lobes]

    D --> H[Differentiable Renderer<br/>sphere-trace + Cook-Torrance GGX]
    E --> H
    F --> H
    G --> H
    H --> I[Rendered pixel color]

    F -. regularizes .-> J[Material Diffusion Prior<br/>frozen, SDS-style loss]
    J -. score .-> F

    F --> K[Ambiguity Penalty<br/>energy conservation]
    G --> K

    C --> L[Uncertainty Estimator<br/>MC-dropout / latent sampling]
    L --> M[Uncertainty map]

    I --> N[Photometric + Mask Losses]
    N --> O[Backprop through renderer<br/>into backbone + heads]
```

## Training stages

```mermaid
flowchart LR
    S0[Synthetic dataset<br/>GT geometry/BRDF/light] --> S1[Stage 1: feed-forward<br/>photometric + mask + eikonal]
    S1 --> S2[Stage 2: + diffusion prior<br/>+ ambiguity penalty + uncertainty]
    S2 --> S3[Fine-tune on real captures<br/>photometric consistency only]
    S3 --> S4[Relighting validation<br/>novel held-out illumination]
    S4 --> S5[Ablation: with vs without prior]
```

## Key design decisions

- **Hexa-Plane over a dense voxel grid or full transformer**: three spatial
  planes (tri-plane decomposition) plus three auxiliary planes modulated by a
  pooled multi-view image feature give cheap cross-view fusion without an
  expensive per-point attention pass, keeping inference in the "seconds on a
  consumer GPU" regime.
- **Decoupled heads**: geometry, material, and illumination are architecturally
  separate consumers of the shared Hexa-Plane feature. This is what lets the
  diffusion prior (Phase 3) attach only to the material head's latent/output
  without disturbing geometry or lighting estimation.
- **SG lighting**: a mixture of Spherical Gaussians gives a compact,
  differentiable, closed-form-integrable low-frequency environment
  representation -- cheap enough to predict from a single feed-forward head,
  and trivial to swap for a *novel* held-out environment at relighting-eval time.
- **SDS-style diffusion-prior loss**: rather than running the material head's
  output through the full diffusion sampling loop at every training step
  (expensive), we use a Score-Distillation-Sampling-style loss: one prior
  forward pass yields a score/denoised-target that nudges the material
  prediction toward the prior's learned manifold.
- **Self-contained differentiable renderer**: implemented in pure PyTorch
  (sphere-traced SDF + analytic Cook-Torrance/GGX) so the project has no hard
  dependency on `nvdiffrast`/`redner`; swapping in a rasterization-based
  renderer for mesh-based assets is a drop-in change since `shade()` only
  consumes per-pixel (point, normal, material) buffers.
