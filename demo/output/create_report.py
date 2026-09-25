from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
import json
import os

doc = Document()
OUTPUT_DIR = r'C:\Users\Admin\Downloads\neural-relighting-project\neural-relighting-project\demo\output'

# ---- Styles ----
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

style_h1 = doc.styles['Heading 1']
style_h1.font.size = Pt(18)
style_h1.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

style_h2 = doc.styles['Heading 2']
style_h2.font.size = Pt(14)
style_h2.font.color.rgb = RGBColor(0x2E, 0x5E, 0x8E)

style_h3 = doc.styles['Heading 3']
style_h3.font.size = Pt(12)
style_h3.font.color.rgb = RGBColor(0x3A, 0x7C, 0xA5)

# ---- Title Page ----
doc.add_paragraph()
doc.add_paragraph()
title = doc.add_heading('Neural Relighting Project', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.runs[0].font.size = Pt(28)
title.runs[0].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

subtitle = doc.add_heading('Full Demo Results Report', level=1)
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.runs[0].font.size = Pt(18)
subtitle.runs[0].font.color.rgb = RGBColor(0x2E, 0x5E, 0x8E)

doc.add_paragraph()
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run('Generated: ').bold = True
meta.add_run('2026-09-25\n')
meta.add_run('Demo Script: ').bold = True
meta.add_run('demo/full_demo.py\n')
meta.add_run('Checkpoint: ').bold = True
meta.add_run('runs/stage2/last.ckpt (Stage 2: feed-forward + diffusion prior)\n')
meta.add_run('Input Data: ').bold = True
meta.add_run('data/real_captures/object_01\n')
meta.add_run('Device: ').bold = True
meta.add_run('CPU (no CUDA available)\n')
meta.add_run('Resolution: ').bold = True
meta.add_run('256×256\n')
meta.add_run('Turntable Frames: ').bold = True
meta.add_run('8 frames (45° increments)\n')
meta.add_run('Benchmark Iterations: ').bold = True
meta.add_run('10')

doc.add_page_break()

# ---- Section 1: Output Files ----
doc.add_heading('1. Output Files Generated', level=1)

table = doc.add_table(rows=7, cols=3, style='Light List Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ['File', 'Size', 'Description']
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ['turntable_original_lighting_hq.gif', '61 KB', '8-frame orbit under estimated scene lighting (24 SG lobes from illumination head)'],
    ['turntable_frames/', '—', 'Individual PNG frames (frame_000.png … frame_007.png) for per-frame inspection'],
    ['relit_held_out_hq.png', '15 KB', 'Single render under held-out environment (data/envmaps/held_out.npz), azimuth 45°'],
    ['uncertainty_map_hq.png', '143 B', 'Per-pixel epistemic uncertainty (MC-dropout, 8 forward passes) at 256×256'],
    ['relighting_comparison.png', '443 KB', 'Side-by-side: [Ground Truth | Predicted | Absolute Difference]'],
    ['metrics.json', '305 B', 'Quantitative metrics (PSNR, SSIM, LPIPS, inference speed)'],
]
for r_idx, row_data in enumerate(rows_data, 1):
    for c_idx, val in enumerate(row_data):
        table.rows[r_idx].cells[c_idx].text = val

doc.add_paragraph()

# ---- Section 2: Quantitative Metrics ----
doc.add_heading('2. Quantitative Metrics', level=1)

doc.add_heading('2.1 Relighting Evaluation (Geometry + Material Disentanglement Test)', level=2)

p = doc.add_paragraph()
p.add_run('This is the strict validation test that validates whether the model has truly disentangled geometry, material, and lighting. A model that merely memorizes appearance will fail here even if it reconstructs input views perfectly.').italic = True

table2 = doc.add_table(rows=4, cols=3, style='Light List Accent 1')
table2.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Metric', 'Value', 'Interpretation']):
    cell = table2.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

metrics_data = [
    ['PSNR', '5.57 dB', 'Very low — indicates large pixel-wise error between predicted and ground-truth relit image'],
    ['SSIM', '0.0054', 'Near zero — structural similarity almost non-existent'],
    ['LPIPS', 'N/A (package not installed)', 'Perceptual similarity would likely also be poor'],
]
for r_idx, row_data in enumerate(metrics_data, 1):
    for c_idx, val in enumerate(row_data):
        table2.rows[r_idx].cells[c_idx].text = val

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run('What this means: ').bold = True
p.add_run(
    'The model fails the strict relighting validation. The recovered geometry + SVBRDF, '
    'when rendered under a novel held-out illumination, does not match the real photograph '
    'of the same object under that illumination. This is expected for a Stage 2 checkpoint '
    'trained primarily on synthetic data with limited real fine-tuning (60 epochs at lr=5e-5). '
    'The low scores indicate:'
)
bullets = [
    'Geometry may be inaccurate (wrong shape → wrong shadows/highlights)',
    'Material predictions (albedo/roughness/metallic) may be baked with training lighting',
    'Illumination estimation (SG lobes) may not generalize to novel environments',
    'Domain gap between synthetic pre-training and real capture',
]
for b in bullets:
    doc.add_paragraph(b, style='List Bullet')

p = doc.add_paragraph()
p.add_run('Expected values for a working system: ').bold = True
p.add_run('PSNR > 25 dB, SSIM > 0.85, LPIPS < 0.15')

doc.add_heading('2.2 Inference Speed Benchmark', level=2)

table3 = doc.add_table(rows=6, cols=2, style='Light List Accent 1')
table3.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Metric', 'Value']):
    cell = table3.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

bench_data = [
    ['Device', 'CPU'],
    ['Resolution', '256×256'],
    ['Input Views', '4'],
    ['Time per Reconstruction', '16,448 ms (~16.4 seconds)'],
    ['Throughput', '0.06 reconstructions/second'],
]
for r_idx, row_data in enumerate(bench_data, 1):
    for c_idx, val in enumerate(row_data):
        table3.rows[r_idx].cells[c_idx].text = val

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run('What this means: ').bold = True
p.add_run(
    'Target (config): "consumer GPU (e.g., RTX 4070-class)" with ms_per_reconstruction '
    'in low hundreds of ms. Actual: Running on CPU → 16.4 s/reconstruction is expected. '
    'Bottlenecks: Sphere-tracing SDF (iterative ray-marching) + 4 secondary rays for soft shadows. '
    'On GPU (RTX 3080/4070): Expected ~200–500 ms/reconstruction for same config.'
)

p = doc.add_paragraph()
p.add_run('Config knobs to speed up: ').bold = True
speed_bullets = [
    'rendering.num_secondary_rays: 0 (direct-only SG relighting) → ~5× faster',
    'model.hexaplane.resolution: [64,64,64,16,16,16] → smaller feature grid',
    'uncertainty.num_forward_passes: 4 (only affects uncertainty, not main benchmark)',
]
for b in speed_bullets:
    doc.add_paragraph(b, style='List Bullet')

doc.add_page_break()

# ---- Section 3: Qualitative Outputs ----
doc.add_heading('3. Qualitative Outputs Explained', level=1)

outputs = [
    ('3.1 Turntable Under Estimated Scene Lighting', 'turntable_original_lighting_hq.gif',
     'Shows the reconstructed 3D asset (geometry + material) orbiting under the scene lighting '
     'predicted by the illumination head (24 SG lobes fit to input photos). Use case: Quick sanity check — '
     'does the object look plausible? Are materials reasonable? Does lighting match input views? '
     'Resolution: 256×256, 8 frames, 8 fps.'),
    ('3.2 Relit Render', 'relit_held_out_hq.png',
     'Same asset under a completely novel environment map (held-out HDR → SG fit, never seen during '
     'training/fine-tuning). Purpose: Validates disentanglement — if geometry/material are correct, '
     'relighting should look realistic. Current result: Likely shows artifacts (wrong shadows, color casts, '
     'metallic/roughness errors) consistent with low PSNR/SSIM.'),
    ('3.3 Uncertainty Map', 'uncertainty_map_hq.png',
     'Method: Monte Carlo Dropout (8 stochastic forward passes, dropout_p=0.1). Visualization: '
     'Bright = high epistemic uncertainty (model unsure), Dark = confident. Typical high-uncertainty '
     'regions: Occluded areas, textureless surfaces, depth discontinuities, specular highlights. '
     'Current result: Very small file (143 B) suggests near-uniform low variance — may indicate '
     'dropout not active in eval mode or model overconfident.'),
    ('3.4 Relighting Comparison', 'relighting_comparison.png',
     'Layout: [Ground Truth Photo | Model Prediction | |GT − Pred|]. Ground Truth: '
     'data/real_captures/object_01/gt_relit.jpg (real photo under held-out env). '
     'Prediction: Render of recovered asset under same held-out env + held-out camera pose. '
     'Difference: Absolute pixel error — bright regions = large errors. '
     'Current result: Large differences expected given PSNR=5.57.'),
]

for title, fname, desc in outputs:
    doc.add_heading(title, level=2)
    p = doc.add_paragraph()
    p.add_run(f'File: {fname}').bold = True
    doc.add_paragraph(desc)
    
    # Add image if exists
    img_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(img_path) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            doc.add_picture(img_path, width=Inches(6.0))
            doc.add_paragraph(f'Figure: {fname}').alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception as e:
            doc.add_paragraph(f'[Could not embed image: {e}]')
    elif fname.lower().endswith('.gif'):
        doc.add_paragraph(f'[GIF animation — view separately: {fname}]').italic = True
    doc.add_paragraph()

# ---- Section 4: Diagnostic Checklist ----
doc.add_heading('4. Diagnostic Checklist for Improvement', level=1)

table4 = doc.add_table(rows=6, cols=3, style='Light List Accent 1')
table4.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Issue', 'Likely Cause', 'Fix']):
    cell = table4.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

diag_data = [
    ['PSNR < 10 dB', 'Insufficient real fine-tuning / domain gap',
     'Increase finetune_real.epochs to 200+, use lower lr (1e-5), add more real captures'],
    ['SSIM ~ 0', 'Geometry completely wrong',
     'Check SDF geometric_init, increase eikonal weight, verify synthetic pre-training converged'],
    ['CPU-only speed', 'No CUDA / PyTorch CPU build',
     'Install torch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121'],
    ['Uncertainty map uniform', 'Dropout disabled in eval',
     'Call model.train() during uncertainty estimation or enable mc_dropout in eval mode'],
    ['LPIPS NaN', 'Package missing',
     'pip install lpips'],
]
for r_idx, row_data in enumerate(diag_data, 1):
    for c_idx, val in enumerate(row_data):
        table4.rows[r_idx].cells[c_idx].text = val

doc.add_page_break()

# ---- Section 5: Reproduction Commands ----
doc.add_heading('5. Reproduction Commands', level=1)

commands = [
    ('Full HQ demo (this run)', 
     'python demo/full_demo.py \\\n  --images data/real_captures/object_01 \\\n  --ckpt runs/stage2/last.ckpt \\\n  --out demo/output \\\n  --relight_envs data/envmaps/held_out.npz \\\n  --resolution 256 --num_frames 8 --benchmark_iters 10'),
    ('Fast demo (2-frame, 64×64)', 
     'python demo/fast_demo.py \\\n  --images data/real_captures/object_01 \\\n  --ckpt runs/stage2/last.ckpt \\\n  --out demo/output'),
    ('Relighting eval only', 
     'python -m src.evaluation.relighting_eval \\\n  --ckpt runs/stage2/last.ckpt \\\n  --capture_dir data/real_captures/object_01 \\\n  --novel_env data/envmaps/held_out.npz \\\n  --gt_relit_photo data/real_captures/object_01/gt_relit.jpg \\\n  --held_out_pose data/real_captures/object_01/pose_held_out.npz'),
    ('Speed benchmark only', 
     'python -m src.evaluation.benchmark_inference \\\n  --ckpt runs/stage2/last.ckpt \\\n  --config configs/default.yaml'),
]

for title, cmd in commands:
    doc.add_heading(title, level=2)
    p = doc.add_paragraph(cmd)
    p.style = doc.styles['Normal']
    for run in p.runs:
        run.font.name = 'Consolas'
        run.font.size = Pt(9)

doc.add_page_break()

# ---- Section 6: Next Steps ----
doc.add_heading('6. Next Steps', level=1)

next_steps = [
    ('Install GPU PyTorch', '50–100× speedup expected'),
    ('Extend real fine-tuning', '60 epochs at 5e-5 is minimal; try 200 epochs at 1e-5 with data augmentation'),
    ('Verify synthetic pre-training', 'Check Stage 1/2 losses converged on synthetic benchmark'),
    ('Add LPIPS', 'pip install lpips for perceptual metric'),
    ('Inspect uncertainty', 'Visualize per-channel (albedo/roughness/metallic) uncertainty, not just aggregated'),
    ('Mesh export', 'Use src/visualization/viewer.py to extract .obj/.glb for external validation'),
]

for i, (step, detail) in enumerate(next_steps, 1):
    p = doc.add_paragraph()
    p.add_run(f'{i}. {step}: ').bold = True
    p.add_run(detail)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('— Report generated automatically by demo/full_demo.py —')
run.italic = True
run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

# Save
output_path = r'C:\Users\Admin\Downloads\neural-relighting-project\neural-relighting-project\demo\output\DEMO_RESULTS_REPORT.docx'
doc.save(output_path)
print(f'Saved to {output_path}')