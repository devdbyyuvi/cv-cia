# Real casual multi-view captures

Expected layout per captured object:

```
data/real_captures/<object_id>/
    images/IMG_0001.jpg ... IMG_000{3..6}.jpg   # 3-6 orbital smartphone photos
    sparse/
        cameras.txt      # COLMAP camera intrinsics
        images.txt       # COLMAP camera poses
        points3D.txt
    masks/IMG_0001.png ...                       # optional foreground masks
```

## How to produce `sparse/` from a photo burst or short orbital video

```bash
# extract frames from a short orbit video (optional, if starting from video)
ffmpeg -i orbit.mov -vf "select='not(mod(n\,10))'" -vsync vfr images/IMG_%04d.jpg

# run COLMAP structure-from-motion to recover poses
colmap automatic_reconstructor \
    --workspace_path data/real_captures/<object_id> \
    --image_path data/real_captures/<object_id>/images \
    --sparse 1 --dense 0

# COLMAP's binary sparse model can be converted to the .txt format expected here:
colmap model_converter \
    --input_path data/real_captures/<object_id>/sparse/0 \
    --output_path data/real_captures/<object_id>/sparse \
    --output_type TXT
```

Alternatively, any app that exports COLMAP-compatible camera poses (e.g.
Polycam, RealityScan) works as a drop-in replacement for the COLMAP CLI step.

`src/data/real_capture_dataset.py` parses `sparse/{cameras,images}.txt`
directly; see that file's docstring for the exact parsing assumptions.
