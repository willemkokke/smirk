import warnings
warnings.filterwarnings("ignore")

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import json
import shutil
import time
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"


def crop_face(frame, landmarks, scale=1.0, image_size=IMAGE_SIZE):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


def detect_input_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    elif ext in VIDEO_EXTENSIONS:
        return 'video'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _to_renderer(tensor, renderer_device):
    """Move tensor to renderer device only if needed."""
    return tensor.cpu() if renderer_device == 'cpu' and tensor.device.type != 'cpu' else tensor


def process_frame(image, args, smirk_encoder, flame, renderer, smirk_generator, face_probabilities):
    """Process a single frame. Returns (grid_numpy, flame_output, outputs, tform_or_None, orig_size)."""
    device = args.device
    renderer_device = 'cpu' if str(device).startswith('mps') else str(device)
    orig_h, orig_w, _ = image.shape
    kpt_mediapipe = run_mediapipe(image)
    tform = None

    if args.crop:
        if kpt_mediapipe is None:
            return None
        kpt_mediapipe = kpt_mediapipe[..., :2]
        tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=IMAGE_SIZE)
        cropped_image = warp(image, tform.inverse, output_shape=(IMAGE_SIZE, IMAGE_SIZE), preserve_range=True).astype(np.uint8)
        cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T
        cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]
    else:
        cropped_image = image
        cropped_kpt_mediapipe = kpt_mediapipe

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    cropped_image = cropped_image.to(device)

    outputs = smirk_encoder(cropped_image)
    flame_output = flame.forward(outputs)

    # pytorch3d rasterizer supports CPU and CUDA but not MPS
    renderer_output = renderer.forward(
        _to_renderer(flame_output['vertices'], renderer_device),
        _to_renderer(outputs['cam'], renderer_device),
        landmarks_fan=_to_renderer(flame_output['landmarks_fan'], renderer_device),
        landmarks_mp=_to_renderer(flame_output['landmarks_mp'], renderer_device))
    rendered_img = renderer_output['rendered_img'].to(device)

    # Build visualization grid
    if args.render_orig:
        if args.crop:
            rendered_img_numpy = (rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(orig_h, orig_w), preserve_range=True).astype(np.uint8)
            rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            rendered_img_orig = F.interpolate(rendered_img, (orig_h, orig_w), mode='bilinear').cpu()
        full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        grid = torch.cat([full_image, rendered_img_orig], dim=3)
    else:
        grid = torch.cat([cropped_image, rendered_img], dim=3)

    # Neural reconstruction via smirk generator
    if args.use_smirk_generator:
        if kpt_mediapipe is None:
            return None

        mask_ratio_mul = 5
        mask_ratio = 0.01
        mask_dilation_radius = 10

        hull_mask = create_mask(cropped_kpt_mediapipe, (IMAGE_SIZE, IMAGE_SIZE))
        rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
        tmask_ratio = mask_ratio * mask_ratio_mul

        npoints, _ = masking_utils.mesh_based_mask_uniform_faces(
            _to_renderer(renderer_output['transformed_vertices'], renderer_device),
            flame_faces=_to_renderer(flame.faces_tensor, renderer_device),
            face_probabilities=face_probabilities,
            mask_ratio=tmask_ratio)
        npoints = npoints.to(device)
        pmask = torch.zeros_like(rendered_mask)
        rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
        rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
        rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()

        for bi in range(npoints.size(0)):
            pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1

        hull_mask = torch.from_numpy(hull_mask).type(dtype=torch.float32).unsqueeze(0).to(args.device)
        extra_points = cropped_image * pmask
        masked_img = masking_utils.masking(cropped_image, hull_mask, extra_points, mask_dilation_radius, rendered_mask=rendered_mask)

        smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)
        reconstructed_img = smirk_generator(smirk_generator_input)

        if args.render_orig:
            if args.crop:
                reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(orig_h, orig_w), preserve_range=True).astype(np.uint8)
                reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            else:
                reconstructed_img_orig = F.interpolate(reconstructed_img, (orig_h, orig_w), mode='bilinear').cpu()
            grid = torch.cat([grid, reconstructed_img_orig], dim=3)
        else:
            grid = torch.cat([grid, reconstructed_img], dim=3)

    grid_numpy = grid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    grid_numpy = grid_numpy.astype(np.uint8)
    grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

    return grid_numpy, flame_output, outputs, tform, (orig_w, orig_h)


def export_scene(out_dir, name, flame_output, outputs, flame, tform, orig_size, frame_index=None):
    """Export OBJ mesh and scene JSON for a single frame."""
    prefix = f"{name}_{frame_index:06d}" if frame_index is not None else name

    # OBJ
    vertices = flame_output['vertices'].squeeze(0).detach().cpu().numpy()
    faces = flame.faces_tensor.detach().cpu().numpy()
    obj_path = os.path.join(out_dir, f"{prefix}.obj")
    with open(obj_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Scene JSON
    cam = outputs['cam'].squeeze(0).detach().cpu().numpy()
    scene_data = {
        'camera': {
            'type': 'orthographic',
            'scale': float(cam[0]),
            'tx': float(cam[1]),
            'ty': float(cam[2]),
            'image_size': IMAGE_SIZE,
        },
        'flame_params': {
            'pose_params': outputs['pose_params'].squeeze(0).detach().cpu().tolist(),
            'shape_params': outputs['shape_params'].squeeze(0).detach().cpu().tolist(),
            'expression_params': outputs['expression_params'].squeeze(0).detach().cpu().tolist(),
            'jaw_params': outputs['jaw_params'].squeeze(0).detach().cpu().tolist(),
            'eyelid_params': outputs['eyelid_params'].squeeze(0).detach().cpu().tolist(),
        },
    }
    if frame_index is not None:
        scene_data['frame'] = frame_index
    if tform is not None:
        scene_data['crop'] = {
            'transform_matrix': tform.params.tolist(),
            'original_image_size': list(orig_size),
        }

    json_path = os.path.join(out_dir, f"{prefix}.json")
    with open(json_path, 'w') as f:
        json.dump(scene_data, f, indent=2)


def run_image(args, smirk_encoder, flame, renderer, smirk_generator, face_probabilities, out_dir):
    image = cv2.imread(args.input_path)
    if image is None:
        print(f"Error: could not read image {args.input_path}")
        return

    t0 = time.time()
    result = process_frame(image, args, smirk_encoder, flame, renderer, smirk_generator, face_probabilities)
    frame_time = time.time() - t0

    if result is None:
        print("Could not find face landmarks. Exiting...")
        return

    grid_numpy, flame_output, outputs, tform, orig_size = result
    name = os.path.splitext(os.path.basename(args.input_path))[0]

    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), grid_numpy)
    print(f"  Inference: {format_time(frame_time)}")

    if args.export_scene:
        t0 = time.time()
        export_scene(out_dir, name, flame_output, outputs, flame, tform, orig_size)
        print(f"  Export:    {format_time(time.time() - t0)}")


def run_video(args, smirk_encoder, flame, renderer, smirk_generator, face_probabilities, out_dir):
    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        print(f"Error: could not open video {args.input_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.splitext(os.path.basename(args.input_path))[0]

    print(f"  {video_width}x{video_height} @ {video_fps:.1f}fps, {total_frames} frames")

    if args.render_orig:
        out_w, out_h = video_width, video_height
    else:
        out_w, out_h = IMAGE_SIZE, IMAGE_SIZE

    out_w *= 3 if args.use_smirk_generator else 2

    video_path = os.path.join(out_dir, f"{name}.mp4")
    cap_out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (out_w, out_h))

    scene_dir = os.path.join(out_dir, "scene") if args.export_scene else None
    if scene_dir:
        os.makedirs(scene_dir, exist_ok=True)

    frame_idx = 0
    skipped = 0
    t_start = time.time()
    frame_times = []

    while True:
        ret, image = cap.read()
        if not ret:
            break

        t0 = time.time()
        result = process_frame(image, args, smirk_encoder, flame, renderer, smirk_generator, face_probabilities)
        dt = time.time() - t0
        frame_times.append(dt)

        if result is None:
            skipped += 1
            frame_idx += 1
            continue

        grid_numpy, flame_output, outputs, tform, orig_size = result
        cap_out.write(grid_numpy)

        if scene_dir:
            export_scene(scene_dir, name, flame_output, outputs, flame, tform, orig_size, frame_index=frame_idx)

        frame_idx += 1

        if total_frames > 0:
            elapsed = time.time() - t_start
            avg_fps = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0
            print(f"\r  Frame {frame_idx}/{total_frames} | {avg_fps:.1f} fps | ETA {format_time(eta)}   ", end="", flush=True)

    cap.release()
    cap_out.release()

    total_time = time.time() - t_start
    avg_ms = np.mean(frame_times) * 1000 if frame_times else 0

    print(f"\r  {frame_idx} frames in {format_time(total_time)} | avg {avg_ms:.0f}ms/frame | {frame_idx/total_time:.1f} fps")
    if skipped:
        print(f"  Skipped {skipped} frames (no face detected)")
    print(f"  Video: {video_path}")
    if scene_dir:
        print(f"  Scene: {scene_dir}/ ({frame_idx} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMIRK unified demo for images and videos')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input image or video')
    parser.add_argument('--device', type=str, default='auto', help='Device: cpu, cuda, mps, or auto (auto uses MPS for video on Apple Silicon, CUDA if available)')
    parser.add_argument('--checkpoint', type=str, default='pretrained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='results', help='Base output directory')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image-to-image translator')
    parser.add_argument('--render_orig', action='store_true', help='Render result at original image/video resolution')
    parser.add_argument('--export_scene', action='store_true', help='Export OBJ mesh and scene JSON (per frame for video)')
    args = parser.parse_args()

    total_start = time.time()

    input_type = detect_input_type(args.input_path)
    input_name = os.path.splitext(os.path.basename(args.input_path))[0]
    out_dir = os.path.join(args.out_path, input_name)

    # Resolve auto device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif input_type == 'video' and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    # pytorch3d rasterizer supports CPU and CUDA, but not MPS
    renderer_device = 'cpu' if args.device.startswith('mps') else args.device

    # Clean existing results
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Initialize models
    print(f"Loading models... (device: {args.device})")
    t0 = time.time()

    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    smirk_generator = None
    face_probabilities = None
    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)
        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k}
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()
        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        flame = FLAME().to(args.device)
        renderer = Renderer().to(renderer_device)

    print(f"  Models loaded in {format_time(time.time() - t0)}")
    print(f"Processing {input_type}: {args.input_path}")

    if input_type == 'image':
        run_image(args, smirk_encoder, flame, renderer, smirk_generator, face_probabilities, out_dir)
    else:
        run_video(args, smirk_encoder, flame, renderer, smirk_generator, face_probabilities, out_dir)

    print(f"Total: {format_time(time.time() - total_start)}")
    print(f"Output: {out_dir}/")
