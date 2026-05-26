import cv2
import numpy as np
from pathlib import Path
import matplotlib.cm as cm


def depth_to_color(depth_map, vmin=None, vmax=None, cmap=cv2.COLORMAP_TURBO):
    """Convert a single-channel depth map to an 8-bit BGR color image.

    - depth_map: numpy array (H,W) float. May contain NaN/Inf for invalid pixels.
    - vmin/vmax: if None, computed from valid positive values in depth_map.
    - cmap: OpenCV colormap id kept for compatibility; viridis is used by default.
    Returns uint8 BGR image.
    """
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim > 2:
        depth = depth[..., 0]
    h, w = depth.shape[:2]
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((h, w, 3), dtype=np.uint8)
    if vmin is None:
        vmin = float(depth[valid].min())
    if vmax is None:
        vmax = float(depth[valid].max())
    if vmax - vmin < 1e-8:
        vmax = vmin + 1.0
    norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    color_rgb = cm.viridis(norm)[:, :, :3]
    color = (color_rgb[:, :, ::-1] * 255.0).astype(np.uint8)
    # set invalid pixels to black
    return np.where(valid[..., None], color, 0)


def conf_to_color(conf_map, vmin=0.0, vmax=1.0, cmap=cv2.COLORMAP_JET):
    """Convert a confidence map to color (BGR uint8)."""
    if conf_map is None:
        return None
    conf = np.asarray(conf_map, dtype=np.float32)
    conf = np.squeeze(conf)
    if conf.ndim > 2:
        conf = conf[..., 0]
    h, w = conf.shape[:2]
    valid = np.isfinite(conf)
    if not np.any(valid):
        return np.zeros((h, w, 3), dtype=np.uint8)
    norm = np.clip((conf - vmin) / (vmax - vmin + 1e-10), 0.0, 1.0)
    norm_uint8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm_uint8, cmap)
    return np.where(valid[..., None], color, 0)


def overlay_depth_on_rgb(rgb_bgr, depth_color_bgr, alpha=0.5):
    """Overlay depth color on top of a BGR image."""
    if rgb_bgr is None or depth_color_bgr is None:
        return None
    if rgb_bgr.ndim == 2:
        rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_GRAY2BGR)
    if rgb_bgr.shape[:2] != depth_color_bgr.shape[:2]:
        depth_color_bgr = cv2.resize(depth_color_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    blended = cv2.addWeighted(rgb_bgr, 1.0 - alpha, depth_color_bgr, alpha, 0)
    return blended


def save_depth_and_conf(depth_map, conf_map, depth_path: str, conf_path: str, vmin=None, vmax=None):
    Path(depth_path).parent.mkdir(parents=True, exist_ok=True)
    Path(conf_path).parent.mkdir(parents=True, exist_ok=True)
    depth_color = depth_to_color(depth_map, vmin=vmin, vmax=vmax)
    conf_color = conf_to_color(conf_map)
    if depth_color is not None:
        cv2.imwrite(depth_path, depth_color)
    if conf_color is not None:
        cv2.imwrite(conf_path, conf_color)
"""
python /data2/dataset/Redal/work_feedforward_3drepo/running/utils/viz_utils.py \
    --input_dir /data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity/big_city_depth/aerial/train/big_high_block_1_depth \
    --output_path /data2/dataset/Redal/work_feedforward_3drepo/assets/big_high_block_1_depth.mp4 \
    --fps 25 \
    --crf 35 \
    --width 640
"""
import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Convert image sequence to video using FFmpeg')
    
    parser.add_argument('--input_dir', type=str, 
                        default='/data2/dataset/Redal/work_feedforward_3drepo/dataset/MOOM/01/imgs',
                        help='Path to the folder containing images')
    
    parser.add_argument('--output_path', type=str, 
                        default='output_video.mp4',
                        help='Path and filename for the output video')
    
    parser.add_argument('--fps', type=int, 
                        default=25, 
                        help='Frames per second of the output video')
    
    parser.add_argument('--crf', type=int, 
                        default=23, 
                        help='Video quality/size factor (0-51). Higher is smaller size.')
    
    parser.add_argument('--width', type=int, 
                        default=-1, 
                        help='Output video width. -1 keeps original size. e.g., 640 or 480.')

    args = parser.parse_args()

    # Get list of image files
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    # Sort files
    sorted_files = sorted(files)
    
    print(f"Found {len(sorted_files)} images. Starting conversion with FFmpeg...")

    # Create a temporary file list for FFmpeg
    temp_list_path = "temp_file_list.txt"
    with open(temp_list_path, 'w') as f:
        for filename in sorted_files:
            safe_path = os.path.join(args.input_dir, filename).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    # Build scale filter
    # If width is specified, use it and force height to be an even number (divisible by 2)
    # 'trunc(ow/a/2)*2' calculates height based on aspect ratio and rounds down to nearest even number
    scale_filter = '-1'
    if args.width > 0:
        scale_filter = f'{args.width}:trunc(ow/a/2)*2'

    # FFmpeg Command
    cmd = [
        'ffmpeg', '-y', 
        '-f', 'concat', '-safe', '0', '-i', temp_list_path,
        '-c:v', 'libx264', 
        '-crf', str(args.crf), 
        '-vf', f'scale={scale_filter}', 
        '-pix_fmt', 'yuv420p',
        '-r', str(args.fps),
        args.output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Video saved to: {args.output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred during video creation: {e}")
    finally:
        if os.path.exists(temp_list_path):
            os.remove(temp_list_path)

if __name__ == '__main__':
    main()