# SMIRK Scene JSON Format

Reference for the `_scene.json` files exported by `demo.py`. Contains everything needed to reconstruct the 3D face mesh and project it onto the source image.

## File Overview

Each run of `demo.py` produces three files in the output directory:

| File | Contents |
|------|----------|
| `<name>.png` | Side-by-side visualization (input / mesh overlay / optional neural reconstruction) |
| `<name>.obj` | FLAME mesh in world space (OBJ format, 1-indexed faces) |
| `<name>_scene.json` | Camera, FLAME parameters, and crop transform (this document) |

## Top-Level Structure

```json
{
  "camera": { ... },
  "flame_params": { ... },
  "crop": { ... }
}
```

---

## `camera`

The renderer uses a **scaled orthographic projection** (also called weak-perspective). There is no lens distortion or perspective foreshortening.

```json
{
  "type": "orthographic",
  "scale": 8.786,
  "tx": -0.005,
  "ty": 0.025,
  "image_size": 224,
  "note": "..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `scale` | float | Uniform scale factor applied to all three axes after translation |
| `tx` | float | X translation added to vertex X before scaling |
| `ty` | float | Y translation added to vertex Y before scaling |
| `image_size` | int | The square image resolution the projection maps into (always 224) |

### Projection Pipeline (world space to pixels)

Given a 3D vertex `(x, y, z)` from the OBJ file, the full projection to pixel coordinates in the **224x224 cropped image** is:

```
Step 1 — Translate XY:
    x' = x + tx
    y' = y + ty
    z' = z

Step 2 — Uniform scale:
    x'' = scale * x'
    y'' = scale * y'
    z'' = scale * z'

Step 3 — Flip Y and Z:
    ndc_x =  x''
    ndc_y = -y''
    ndc_z = -z''

Step 4 — NDC to pixel (PyTorch3D convention):
    pixel_x = image_size / 2  -  ndc_x * image_size / 2
    pixel_y = image_size / 2  -  ndc_y * image_size / 2
```

Or as a single formula:

```
pixel_x = (image_size / 2) * (1 - scale * (x + tx))
pixel_y = (image_size / 2) * (1 + scale * (y + ty))
```

**Note on depth:** The Z value is only used for occlusion sorting during rasterization. The renderer offsets it by +10 internally to ensure positive depth. If you need depth in your importer, use `ndc_z` for relative ordering.

### Coordinate System

- **FLAME world space** is right-handed: +X is left (from the subject's perspective), +Y is up, +Z is toward the camera.
- After projection, pixel (0, 0) is the **top-left** corner of the 224x224 image.

---

## `flame_params`

All parameters needed to regenerate the mesh from the FLAME model. These are the raw encoder outputs.

```json
{
  "pose_params": [rx, ry, rz],
  "shape_params": [300 floats],
  "expression_params": [50 floats],
  "jaw_params": [jaw_open, jaw_lateral_1, jaw_lateral_2],
  "eyelid_params": [left_eyelid, right_eyelid]
}
```

| Field | Length | Description |
|-------|--------|-------------|
| `pose_params` | 3 | Global head rotation as an **axis-angle** (Rodrigues) vector. The rotation angle is the vector's magnitude in radians; the direction is the rotation axis. |
| `shape_params` | 300 | FLAME identity shape coefficients (PCA basis). These define the person's face shape. |
| `expression_params` | 50 | FLAME expression coefficients (PCA basis). These capture the current facial expression. |
| `jaw_params` | 3 | Jaw pose as axis-angle. `[0]` is jaw opening (clamped >= 0), `[1:2]` are lateral jaw movement (clamped to +/- 0.2). |
| `eyelid_params` | 2 | Eyelid closure blend weights, `[left, right]`, each in range [0, 1]. 0 = open, 1 = fully closed. |

### How FLAME Builds the Mesh

1. Start from the FLAME template mesh (5023 vertices)
2. Add shape and expression blendshapes: `v = template + shapedirs @ [shape_params, expression_params]`
3. Apply linear blend skinning (LBS) with the full pose: `[pose_params (3), neck_pose (3), jaw_params (3), eye_pose (6)]`
   - `neck_pose` and `eye_pose` default to zero in SMIRK
4. Add eyelid offsets: `v += left_eyelid * l_eyelid_delta + right_eyelid * r_eyelid_delta`

The output is the vertex positions in the OBJ file.

### Using in Maya

For a Maya importer, you have two options:

**Option A (simple):** Import the OBJ directly and use the camera parameters to set up projection. This is sufficient for rendering the mesh over the image.

**Option B (full FLAME rig):** Use the FLAME parameters to drive a FLAME rig in Maya. This gives you editable shape/expression controls. You would need the FLAME model loaded in Maya (via the official FLAME Maya plugin or a custom setup) and apply `shape_params` and `expression_params` as blendshape weights, `pose_params` as the global rotation, and `jaw_params` as the jaw joint rotation.

---

## `crop`

Only present when `--crop` was used. Describes the affine transform between the original image and the 224x224 crop that SMIRK operates on.

```json
{
  "transform_matrix": [[3x3 matrix]],
  "original_image_size": [width, height],
  "note": "..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `transform_matrix` | 3x3 float array | Affine similarity transform (rotation, uniform scale, translation) |
| `original_image_size` | [int, int] | Original image dimensions as `[width, height]` |

### Using the Transform

The matrix `M` maps **original image coordinates to crop coordinates**:

```
[crop_x]       [orig_x]
[crop_y] = M @ [orig_y]
[  1   ]       [  1   ]
```

To go from **crop pixel coordinates back to the original image**, invert the matrix:

```
[orig_x]            [crop_x]
[orig_y] = M^(-1) @ [crop_y]
[  1   ]            [  1   ]
```

### Full Pipeline: Mesh to Original Image Pixels

To overlay the mesh on the original (uncropped) image:

1. Project 3D vertex to crop pixels using the camera (see above)
2. Apply the inverse crop transform to get original image pixels:

```
crop_px = (image_size / 2) * (1 - scale * (x + tx))
crop_py = (image_size / 2) * (1 + scale * (y + ty))

[orig_px]              [crop_px]
[orig_py] = M_inv  @   [crop_py]
[  1    ]              [  1    ]
```

---

## Example: Complete Vertex Projection (Python)

```python
import json
import numpy as np

with open("results/test_image2_scene.json") as f:
    scene = json.load(f)

cam = scene["camera"]
s, tx, ty = cam["scale"], cam["tx"], cam["ty"]
sz = cam["image_size"]  # 224

# Load a vertex from the OBJ (e.g., nose tip)
vx, vy, vz = 0.0, -0.02, 0.05  # example

# Project to crop pixels
crop_px = (sz / 2.0) * (1.0 - s * (vx + tx))
crop_py = (sz / 2.0) * (1.0 + s * (vy + ty))

# If cropped, map back to original image
if "crop" in scene:
    M = np.array(scene["crop"]["transform_matrix"])
    M_inv = np.linalg.inv(M)
    orig = M_inv @ np.array([crop_px, crop_py, 1.0])
    orig_px, orig_py = orig[0], orig[1]
```

## OBJ File

Standard Wavefront OBJ with vertices (`v`) and triangular faces (`f`). Faces are 1-indexed. No materials, UVs, or normals are exported. The mesh is the full FLAME head topology (5023 vertices, 9976 faces) in world space, before any camera projection.
