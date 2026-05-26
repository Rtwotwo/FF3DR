"""
python -m running.utils.depth_utils \
  --input_dir /data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS/test/area3/depths/3 \
  --output_dir /data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS/test/area3/depths_viz/3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
	import OpenEXR  # type: ignore
	import Imath  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	OpenEXR = None
	Imath = None

from running.utils.viz_utils import depth_to_color


DEPTH_EXTENSIONS = {".pfm", ".exr", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def read_pfm(path: str) -> Tuple[np.ndarray, float]:
	with open(path, "rb") as fh:
		header = fh.readline().decode("ascii", errors="ignore").rstrip()
		if header not in {"PF", "Pf"}:
			raise ValueError(f"Invalid PFM header: {header}")

		dims = fh.readline().decode("ascii", errors="ignore").strip()
		while dims.startswith("#"):
			dims = fh.readline().decode("ascii", errors="ignore").strip()
		width, height = map(int, dims.split())

		scale = float(fh.readline().decode("ascii", errors="ignore").strip())
		endian = "<" if scale < 0 else ">"
		scale = abs(scale)

		data = np.fromfile(fh, endian + "f")
		channels = 3 if header == "PF" else 1
		expected = width * height * channels
		if data.size != expected:
			raise ValueError(f"PFM size mismatch: expected {expected}, got {data.size}")

		shape = (height, width, 3) if channels == 3 else (height, width)
		data = np.reshape(data, shape)
		data = np.flipud(data)
		return data.astype(np.float32), scale


def read_exr_single_channel(path: str, channel_name: str = "Y") -> np.ndarray:
	if OpenEXR is None or Imath is None:
		raise ImportError("OpenEXR/Imath is required to read EXR depth maps")

	exr_file = OpenEXR.InputFile(path)
	header = exr_file.header()
	data_window = header["dataWindow"]
	width = data_window.max.x - data_window.min.x + 1
	height = data_window.max.y - data_window.min.y + 1
	pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
	depth_bytes = exr_file.channel(channel_name, pixel_type)
	depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(height, width)
	return depth.copy()


def read_depth_map(path: str) -> np.ndarray:
	suffix = Path(path).suffix.lower()
	if suffix == ".pfm":
		depth, _ = read_pfm(path)
		return depth
	if suffix == ".exr":
		try:
			return read_exr_single_channel(path)
		except Exception:
			if OpenEXR is None:
				raise
			exr_file = OpenEXR.InputFile(path)
			header = exr_file.header()
			data_window = header["dataWindow"]
			width = data_window.max.x - data_window.min.x + 1
			height = data_window.max.y - data_window.min.y + 1
			pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
			channels = list(header["channels"].keys())
			if not channels:
				raise ValueError(f"EXR has no channels: {path}")
			channel = sorted(channels)[0]
			depth_bytes = exr_file.channel(channel, pixel_type)
			depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(height, width)
			return depth.copy()
	if suffix == ".npy":
		return np.load(path).astype(np.float32)

	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise ValueError(f"Failed to read depth image: {path}")
	if img.ndim == 3:
		img = img[..., 0]
	if img.dtype == np.uint16:
		img = img.astype(np.float32)
	else:
		img = img.astype(np.float32)
	return img


def iter_depth_files(input_dir: Path, recursive: bool = True) -> Iterable[Path]:
	pattern = "**/*" if recursive else "*"
	for path in sorted(input_dir.glob(pattern)):
		if path.is_file() and path.suffix.lower() in DEPTH_EXTENSIONS:
			yield path


def visualize_depth_file(
	src_path: Path,
	dst_path: Path,
	vmin: Optional[float] = None,
	vmax: Optional[float] = None,
) -> Path:
	depth = read_depth_map(str(src_path))
	depth = np.asarray(depth, dtype=np.float32)
	if depth.ndim > 2:
		depth = depth[..., 0]

	color = depth_to_color(depth, vmin=vmin, vmax=vmax)
	dst_path.parent.mkdir(parents=True, exist_ok=True)
	if color is None:
		raise ValueError(f"Failed to convert depth to color: {src_path}")
	cv2.imwrite(str(dst_path), color)
	return dst_path


def visualize_depth_folder(
	input_dir: str,
	output_dir: str,
	recursive: bool = True,
	preserve_tree: bool = True,
	suffix: str = "_depth.png",
	vmin: Optional[float] = None,
	vmax: Optional[float] = None,
	progress_every: int = 50,
) -> int:
	input_path = Path(input_dir)
	output_path = Path(output_dir)
	if not input_path.exists():
		raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

	sources = list(iter_depth_files(input_path, recursive=recursive))
	total = len(sources)
	print(f"[INFO] Start depth visualization: total_files={total}", flush=True)
	if total == 0:
		print(f"[WARN] No supported depth files found in: {input_dir}", flush=True)
		return 0

	count = 0
	for src in sources:
		rel = src.relative_to(input_path) if preserve_tree else Path(src.name)
		dst_name = src.stem + suffix
		dst = output_path / rel.parent / dst_name
		try:
			visualize_depth_file(src, dst, vmin=vmin, vmax=vmax)
		except Exception as exc:
			print(f"[WARN] Skip {src}: {exc}", flush=True)
			continue
		count += 1
		if progress_every > 0 and (count % progress_every == 0 or count == total):
			print(f"[INFO] Progress: {count}/{total}", flush=True)
	return count


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Visualize all depth maps in a folder with a unified color style."
	)
	parser.add_argument("--input_dir", type=str, required=True, help="Folder containing depth maps")
	parser.add_argument("--output_dir", type=str, required=True, help="Folder to save visualized PNGs")
	parser.add_argument("--no_recursive", action="store_true", help="Do not scan subfolders recursively")
	parser.add_argument("--no_preserve_tree", action="store_true", help="Do not preserve relative folder structure")
	parser.add_argument("--suffix", type=str, default="_depth.png", help="Output filename suffix")
	parser.add_argument("--vmin", type=float, default=None, help="Manual min depth for color mapping")
	parser.add_argument("--vmax", type=float, default=None, help="Manual max depth for color mapping")
	parser.add_argument("--progress_every", type=int, default=50, help="Print progress every N images")
	return parser


def main() -> None:
	parser = build_argparser()
	args = parser.parse_args()
	count = visualize_depth_folder(
		input_dir=args.input_dir,
		output_dir=args.output_dir,
		recursive=not args.no_recursive,
		preserve_tree=not args.no_preserve_tree,
		suffix=args.suffix,
		vmin=args.vmin,
		vmax=args.vmax,
		progress_every=args.progress_every,
	)
	print(f"[INFO] Visualized {count} depth maps from {args.input_dir} -> {args.output_dir}")


if __name__ == "__main__":
	main()
