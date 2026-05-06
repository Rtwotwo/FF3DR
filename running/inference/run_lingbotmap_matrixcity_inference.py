"""LingBot-MAP: streaming 3D reconstruction for MatrixCity (big_city & small_city).

Usage:
    # Single block reconstruction with result saving
    python run_lingbotmap_matrixcity_inference.py \
        --model_path ./weights/lingbot-map/lingbot-map.pt \
        --image_folder ./dataset/MatrixCity/small_city/aerial/train/block_1/ \
        --output_path ./exp/matrixcity/run_matrixcity_lingbotmap_train_block_1

    # Streaming inference with keyframe KV caching
    python run_lingbotmap_matrixcity_inference.py \
        --model_path ./weights/lingbot-map/lingbot-map.pt \
        --image_folder /path/to/images/ --mode streaming --keyframe_interval 6 \
        --output_path ./exp/matrixcity/output

    # Windowed inference (for very long sequences, >500 frames)
    python run_lingbotmap_matrixcity_inference.py \
        --model_path ./weights/lingbot-map/lingbot-map.pt \
        --image_folder /path/to/images/ --mode windowed --window_size 64 \
        --output_path ./exp/matrixcity/output

    # Visualization only (no saving)
    python run_lingbotmap_matrixcity_inference.py \
        --model_path ./weights/lingbot-map/lingbot-map.pt \
        --image_folder /path/to/images/ --vis_only
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Must be set before `import torch` / any CUDA init. Reduces the reserved-vs-allocated
# memory gap by letting the caching allocator grow segments on demand instead of
# pre-reserving fixed-size blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "models").is_dir() and (p / "configs").is_dir():
            return p
    return start.parent


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MODELS_DIR = _REPO_ROOT / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from models.lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from models.lingbot_map.utils.geometry import closed_form_inverse_se3_general
from models.lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8):
    """Load images from folder or video and preprocess into a tensor.

    Returns:
        (images, paths, resolved_image_folder): preprocessed tensor, file paths,
        and the folder containing the source images (for sky mask caching etc.).
    """
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using canonical crop mode")
    return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# torch.compile (opt-in via --compile)
# =============================================================================

def compile_model(model):
    """Compile hot, fixed-shape modules with mode="reduce-overhead".

    Mirrors the targets in gct_profile.py:compile_model. Unlike the profile script,
    `model.point_head` is **kept** — the demo needs world_points for visualization.
    """
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")


def _warm_streaming(model, images, scale_frames, warm_stream_n, dtype, passes=1):
    """Drive `clean_kv_cache → Phase 1 → N streaming forwards` `passes` times.

    Warmup inputs are sliced from the already-preprocessed `images` tensor, so their
    spatial shape matches what real inference will feed — this is what makes the
    captured CUDA graphs reusable (reduce-overhead mode keys on shape).
    """
    # images: [S, 3, H, W] on device already; slice and add batch dim.
    warm_scale = images[:scale_frames].unsqueeze(0).to(dtype)
    warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(dtype)

    for _ in range(passes):
        model.clean_kv_cache()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(
                warm_scale,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        for i in range(warm_stream_n):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(
                    warm_stream[:, i:i + 1],
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Wipe warmup KV so real inference_streaming starts clean (it also calls
    # clean_kv_cache internally, but this is defensive + makes intent obvious).
    model.clean_kv_cache()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


def _write_ply_header(f, num_vertices):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    f.write("\n".join(header).encode() + b"\n")


def _write_ply_batch(f, points, colors):
    structured = np.zeros(
        len(points),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ],
    )
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    structured["red"] = colors[:, 0]
    structured["green"] = colors[:, 1]
    structured["blue"] = colors[:, 2]
    f.write(structured.tobytes())


def _save_pointcloud_ply(points, colors, confs, output_path, conf_threshold, sample_ratio=1.0):
    if points.ndim == 2:
        points = points[np.newaxis, ...]
        colors = colors[np.newaxis, ...]
        confs = confs[np.newaxis, ...]

    all_pts = []
    all_cls = []
    for i in range(points.shape[0]):
        pts = points[i].reshape(-1, 3).astype(np.float32)
        cls = colors[i].reshape(-1, 3).astype(np.uint8)
        cfs = confs[i].reshape(-1).astype(np.float32)
        mask = (cfs >= conf_threshold) & (cfs > 1e-5)
        all_pts.append(pts[mask])
        all_cls.append(cls[mask])

    valid_pts = np.concatenate(all_pts, axis=0)
    valid_cls = np.concatenate(all_cls, axis=0)

    if len(valid_pts) == 0:
        with open(output_path, "wb") as f:
            _write_ply_header(f, 0)
        return

    if 0 < sample_ratio < 1.0:
        num_samples = int(len(valid_pts) * sample_ratio)
        indices = np.random.choice(len(valid_pts), num_samples, replace=False)
        valid_pts = valid_pts[indices]
        valid_cls = valid_cls[indices]

    with open(output_path, "wb") as f:
        _write_ply_header(f, len(valid_pts))
        _write_ply_batch(f, valid_pts, valid_cls)

    print(f"Saved point cloud with {len(valid_pts)} points to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
                        help="streaming: frame-by-frame with KV cache; windowed: overlapping windows for long sequences")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Streaming only. Every N-th frame after scale frames is kept as a keyframe. 1 = every frame. "
            "If unset, auto-selected: 1 when num_frames <= 320, else ceil(num_frames / 320).",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4,
                        help="Camera head iterative-refinement steps. Default 4; set 1 for faster inference "
                            "(skips 3 refinement passes at a small accuracy cost).")
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile hot modules (reduce-overhead) with a CUDA-graph warmup. "
                            "Streaming mode only; ~5 FPS faster at 518x378. Adds ~30-60 s warmup time.")
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        help="Offload per-frame predictions to CPU during inference to cut GPU peak memory. "
            "Use --no-offload_to_cpu to keep outputs on GPU.",
    )
    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16, help="Overlap between windows")

    # Output / saving
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output directory for saving reconstruction results (PLY, npy). "
                            "If not set, only visualization is launched.")
    parser.add_argument("--conf_threshold_coef", type=float, default=0.75,
                        help="Confidence threshold coefficient for PLY saving. "
                            "conf_threshold = mean(confs) * coef")
    parser.add_argument("--sample_ratio", type=float, default=0.015,
                        help="Point cloud sampling ratio for PLY saving (0 < ratio <= 1.0)")
    parser.add_argument("--save_glb", action="store_true", default=False,
                        help="Also export GLB file (requires trimesh)")
    parser.add_argument("--no_vis", action="store_true", default=False,
                        help="Skip visualization (useful for batch processing)")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export stride-sampled, resized/cropped images to this folder")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
    )

    # Export preprocessed images if requested
    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        print(f"Exporting {images.shape[0]} preprocessed images to {args.export_preprocessed}...")
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        print(f"Exported to {args.export_preprocessed}")

    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    # Pick inference dtype; autocast still runs for the ops that need fp32 (e.g. LayerNorm).
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Cast the aggregator (DINOv2-style trunk) to the inference dtype to remove the
    # redundant fp32 master weight copy + autocast bf16 weight cache (~2-3 GB saved,
    # no measurable quality change). gct_base._predict_* upcasts inputs to fp32 and
    # runs each head under `autocast(enabled=False)`, so camera/depth/point heads
    # keep fp32 weights automatically.
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}")
    print(f"Mode: {args.mode}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU mem after load: "
            f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
            print(
                f"Auto-selected --keyframe_interval={args.keyframe_interval} "
                f"(num_frames={num_frames} > 320)."
            )
        else:
            args.keyframe_interval = 1

    if args.mode != "streaming" and args.keyframe_interval != 1:
        print("Warning: --keyframe_interval only applies to --mode streaming. Ignoring it for windowed inference.")
        args.keyframe_interval = 1
    elif args.mode == "streaming" and args.keyframe_interval > 1:
        print(
            f"Keyframe streaming enabled: interval={args.keyframe_interval} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )

    # ── Optional: torch.compile + CUDA-graph warmup (streaming only) ────────
    if args.compile:
        if args.mode != "streaming":
            print(
                f"--compile only applies to --mode streaming (got {args.mode!r}); "
                "skipping compile."
            )
        else:
            scale_for_warm = min(args.num_scale_frames, num_frames)
            warm_stream_n = min(10, max(1, num_frames - scale_for_warm))
            print(f"Warmup eager (scale + {warm_stream_n} streaming)...")
            t_warm = time.time()
            _warm_streaming(model, images, scale_for_warm, warm_stream_n, dtype, passes=1)
            print(f"  eager warmup: {time.time() - t_warm:.1f}s")

            print("Compiling hot modules...")
            compile_model(model)

            # 3 passes under compile: 1st captures CUDA graphs, 2nd/3rd replay so
            # the caching allocator / graph-address map converge on the state the
            # real inference will see. See gct_profile.py:302-306 for rationale.
            print("Warmup compiled (3x dress rehearsal)...")
            t_warm = time.time()
            _warm_streaming(model, images, scale_for_warm, warm_stream_n, dtype, passes=3)
            print(f"  compiled warmup: {time.time() - t_warm:.1f}s")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                output_device=output_device,
            )

    print(f"Inference done in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(
            f"GPU peak during inference: "
            f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB "
            f"(reserved peak {torch.cuda.max_memory_reserved()/1e9:.2f} GB)"
        )

    # ── Post-process ─────────────────────────────────────────────────────────
    if args.offload_to_cpu:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]  # already CPU
    else:
        images_for_post = images

    predictions, images_cpu = postprocess(predictions, images_for_post)

    # ── Save results ─────────────────────────────────────────────────────────
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        pcd_dir = os.path.join(args.output_path, "tmp_predictions_pcd")
        os.makedirs(pcd_dir, exist_ok=True)

        vis_pred = prepare_for_visualization(predictions, images_cpu)

        world_points = vis_pred.get("world_points")
        world_points_conf = vis_pred.get("world_points_conf", vis_pred.get("depth_conf"))
        extrinsic = vis_pred.get("extrinsic")
        intrinsic = vis_pred.get("intrinsic")
        depth = vis_pred.get("depth")
        vis_images = vis_pred.get("images")

        if world_points is None and depth is not None and extrinsic is not None and intrinsic is not None:
            from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
            print("world_points not in model output, computing from depth + cameras...")
            world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
            vis_pred["world_points"] = world_points
            if world_points_conf is None:
                world_points_conf = vis_pred.get("depth_conf")
                if world_points_conf is not None:
                    vis_pred["world_points_conf"] = world_points_conf
            print(f"Computed world_points: shape={world_points.shape}")

        save_dict = {}
        if depth is not None:
            save_dict["depth"] = depth
        if world_points_conf is not None:
            save_dict["conf"] = world_points_conf
        if extrinsic is not None:
            save_dict["extrinsics"] = extrinsic
        if intrinsic is not None:
            save_dict["intrinsics"] = intrinsic
        if world_points is not None:
            save_dict["world_points"] = world_points
        if vis_images is not None:
            save_dict["images"] = vis_images

        npy_path = os.path.join(args.output_path, "predictions.npy")
        np.save(npy_path, save_dict, allow_pickle=True)
        print(f"Saved predictions: {npy_path}")

        ply_saved = False
        if world_points is not None and world_points_conf is not None:
            try:
                if vis_images is not None:
                    if vis_images.ndim == 4 and vis_images.shape[-1] == 3:
                        colors = (vis_images * 255.0).clip(0, 255).astype(np.uint8)
                    elif vis_images.ndim == 4 and vis_images.shape[1] == 3:
                        colors = (vis_images.transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        colors = None
                else:
                    colors = None

                if colors is None:
                    colors = np.zeros_like(world_points, dtype=np.uint8)

                conf_threshold = np.mean(world_points_conf) * args.conf_threshold_coef
                ply_path = os.path.join(pcd_dir, "reconstruction.ply")
                _save_pointcloud_ply(
                    points=world_points,
                    colors=colors,
                    confs=world_points_conf,
                    output_path=ply_path,
                    conf_threshold=conf_threshold,
                    sample_ratio=args.sample_ratio,
                )
                print(f"Saved point cloud: {ply_path}")
                ply_saved = True
            except Exception as e:
                print(f"[WARN] PLY saving failed: {e}")
                print("  NPY predictions are still saved.")

        if ply_saved and os.path.exists(npy_path):
            os.remove(npy_path)
            print(f"Removed intermediate file: {npy_path}")

        if args.save_glb:
            try:
                from lingbot_map.vis import predictions_to_glb
                glb_path = os.path.join(args.output_path, "reconstruction.glb")
                scene = predictions_to_glb(vis_pred, conf_thres=args.conf_threshold)
                scene.export(glb_path)
                print(f"Saved GLB: {glb_path}")
            except ImportError:
                print("trimesh not installed. Skipping GLB export.")

        print(f"All results saved to: {args.output_path}")

    # ── Visualize ────────────────────────────────────────────────────────────
    if not args.no_vis:
        try:
            from lingbot_map.vis import PointCloudViewer
            viewer = PointCloudViewer(
                pred_dict=prepare_for_visualization(predictions, images_cpu),
                port=args.port,
                vis_threshold=args.conf_threshold,
                downsample_factor=args.downsample_factor,
                point_size=args.point_size,
                mask_sky=args.mask_sky,
                image_folder=resolved_image_folder,
                sky_mask_dir=args.sky_mask_dir,
                sky_mask_visualization_dir=args.sky_mask_visualization_dir,
            )
            print(f"3D viewer at http://localhost:{args.port}")
            viewer.run()
        except ImportError:
            print("viser not installed. Install with: pip install lingbot-map[vis]")
            print(f"Predictions contain keys: {list(predictions.keys())}")
    else:
        print("Visualization skipped (--no_vis).")


if __name__ == "__main__":
    main()
