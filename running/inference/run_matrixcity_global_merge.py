import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _find_repo_root(start):
    for p in [start, *start.parents]:
        if (p / "models").is_dir() and (p / "loop_utils").is_dir() and (p / "configs").is_dir():
            return p
    return start.parent


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from running.utils.config_utils import load_config
from loop_utils.sim3utils import (
    compute_alignment_error,
    merge_ply_files,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)
from loop_utils.alignment_torch import apply_sim3_direct_torch
from loop_utils.loop_detector import LoopDetector


def discover_block_outputs(base_output_dir, city_size, split, model_name):
    pattern = os.path.join(
        base_output_dir,
        "run_matrixcity_{}_{}_{}_*".format(model_name, city_size, split),
    )
    dirs = sorted(glob.glob(pattern))
    blocks = []
    for d in dirs:
        merged = os.path.join(d, "reconstruction_merged.ply")
        aligned_dir = os.path.join(d, "tmp_predictions_aligned")
        if os.path.isfile(merged) or os.path.isdir(aligned_dir):
            block_name = os.path.basename(d).split(split + "_")[-1]
            blocks.append({"name": block_name, "output_dir": d})
    return blocks


def load_block_point_data(block_info, max_chunks=None):
    aligned_dir = os.path.join(block_info["output_dir"], "tmp_predictions_aligned")
    if not os.path.isdir(aligned_dir):
        logger.warning("[WARN] No aligned data in %s", block_info["output_dir"])
        return None
    chunk_files = sorted(glob.glob(os.path.join(aligned_dir, "chunk_*.npy")))
    if max_chunks is not None and max_chunks > 0:
        chunk_files = chunk_files[:max_chunks]
    all_points = []
    all_conf = []
    all_colors = []
    for cf in chunk_files:
        try:
            data = np.load(cf, allow_pickle=True).item()
            wp = data.get("world_points", None)
            cf_val = data.get("conf", None)
            img = data.get("images", None)
            if wp is None or cf_val is None:
                continue
            all_points.append(wp)
            all_conf.append(cf_val)
            if img is not None:
                if img.dtype != np.uint8:
                    if np.max(img) <= 1.0:
                        img = (img * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img = img.clip(0, 255).astype(np.uint8)
                all_colors.append(img)
        except Exception as e:
            logger.warning("[WARN] Failed to load %s: %s", cf, str(e))
    if len(all_points) == 0:
        return None
    return {
        "world_points": np.concatenate(all_points, axis=0),
        "conf": np.concatenate(all_conf, axis=0),
        "images": np.concatenate(all_colors, axis=0) if all_colors else None,
    }


def extract_block_descriptors(block_info, config, sample_stride=10):
    from PIL import Image as PILImage
    image_cache_dir = os.path.join(block_info["output_dir"], "block_images_for_vpr")
    if not os.path.isdir(image_cache_dir):
        os.makedirs(image_cache_dir, exist_ok=True)
        unaligned_dir = os.path.join(block_info["output_dir"], "tmp_predictions_unaligned")
        chunk_files = sorted(glob.glob(os.path.join(unaligned_dir, "chunk_*.npy")))
        count = 0
        for cf in chunk_files:
            try:
                data = np.load(cf, allow_pickle=True).item()
                imgs = data.get("processed_images", data.get("images", None))
                if imgs is None:
                    continue
                if imgs.ndim == 4:
                    for i in range(0, imgs.shape[0], sample_stride):
                        img = imgs[i]
                        if img.dtype != np.uint8:
                            img = img.clip(0, 255).astype(np.uint8)
                        PILImage.fromarray(img).save(
                            os.path.join(image_cache_dir, "frame_{:06d}.png".format(count))
                        )
                        count += 1
            except Exception as e:
                logger.warning("[WARN] Failed to extract images from %s: %s", cf, str(e))
    if not os.listdir(image_cache_dir):
        logger.warning("[WARN] No images extracted for VPR in block %s", block_info["name"])
        return None, []
    detector = LoopDetector(image_cache_dir, config=config)
    detector.load_model()
    detector.get_image_paths()
    if len(detector.image_paths) == 0:
        return None, []
    detector.extract_descriptors()
    return detector.descriptors, detector.image_paths


def find_cross_block_matches(detectors_info, similarity_threshold=0.80, min_match_count=3):
    import faiss
    block_pairs = []
    for i in range(len(detectors_info)):
        for j in range(i + 1, len(detectors_info)):
            desc_i = detectors_info[i]["descriptors"]
            desc_j = detectors_info[j]["descriptors"]
            if desc_i is None or desc_j is None:
                continue
            desc_i_np = desc_i.numpy() if torch.is_tensor(desc_i) else desc_i
            desc_j_np = desc_j.numpy() if torch.is_tensor(desc_j) else desc_j
            embed_size = desc_i_np.shape[1]
            index_j = faiss.IndexFlatIP(embed_size)
            index_j.add(desc_j_np)
            similarities, indices = index_j.search(desc_i_np, 5)
            matches = []
            for qi in range(len(desc_i_np)):
                for k in range(similarities.shape[1]):
                    sim = similarities[qi, k]
                    nn_idx = indices[qi, k]
                    if sim > similarity_threshold:
                        matches.append((qi, int(nn_idx), float(sim)))
            if len(matches) >= min_match_count:
                avg_sim = np.mean([m[2] for m in matches])
                block_pairs.append({
                    "block_i": i,
                    "block_j": j,
                    "matches": matches,
                    "avg_similarity": float(avg_sim),
                    "num_matches": len(matches),
                })
                logger.info(
                    "[INFO] Cross-block match: block_%d <-> block_%d, matches=%d, avg_sim=%.4f",
                    i, j, len(matches), avg_sim,
                )
    block_pairs.sort(key=lambda x: x["avg_similarity"], reverse=True)
    return block_pairs


def compute_block_transform_via_points(block_i_data, block_j_data, config):
    pts_i = block_i_data["world_points"]
    conf_i = block_i_data["conf"]
    pts_j = block_j_data["world_points"]
    conf_j = block_j_data["conf"]
    n_i = pts_i.shape[0]
    n_j = pts_j.shape[0]
    tail_n = max(1, min(n_i, n_j) // 4)
    point_map_i = pts_i[-tail_n:]
    conf_map_i = conf_i[-tail_n:]
    point_map_j = pts_j[:tail_n]
    conf_map_j = conf_j[:tail_n:]
    if point_map_i.ndim == 3:
        point_map_i = point_map_i[np.newaxis, ...]
        conf_map_i = conf_map_i[np.newaxis, ...]
    if point_map_j.ndim == 3:
        point_map_j = point_map_j[np.newaxis, ...]
        conf_map_j = conf_map_j[np.newaxis, ...]
    try:
        conf_threshold = min(np.median(conf_i), np.median(conf_j)) * 0.1
        s, R, t = weighted_align_point_maps(
            point_map_i, conf_map_i, point_map_j, conf_map_j,
            conf_threshold=conf_threshold, config=config, precompute_scale=None,
        )
        err = float(compute_alignment_error(
            point_map_i, conf_map_i, point_map_j, conf_map_j, conf_threshold, s, R, t
        ))
        logger.info("[INFO] Block transform: s=%.6f, err=%.4f", s, err)
        return s, R, t, err
    except Exception as e:
        logger.warning("[WARN] Block alignment failed: %s", str(e))
        return None


def compute_block_transform_via_icp(block_i_data, block_j_data, config, voxel_size=0.5, max_iter=100):
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("[WARN] Open3D not available, skipping ICP alignment")
        return None
    pts_i = block_i_data["world_points"].reshape(-1, 3).astype(np.float64)
    conf_i = block_i_data["conf"].reshape(-1)
    pts_j = block_j_data["world_points"].reshape(-1, 3).astype(np.float64)
    conf_j = block_j_data["conf"].reshape(-1)
    conf_thresh_i = np.mean(conf_i) * 0.75
    conf_thresh_j = np.mean(conf_j) * 0.75
    mask_i = conf_i > conf_thresh_i
    mask_j = conf_j > conf_thresh_j
    pts_i_valid = pts_i[mask_i]
    pts_j_valid = pts_j[mask_j]
    if len(pts_i_valid) > 500000:
        idx = np.random.choice(len(pts_i_valid), 500000, replace=False)
        pts_i_valid = pts_i_valid[idx]
    if len(pts_j_valid) > 500000:
        idx = np.random.choice(len(pts_j_valid), 500000, replace=False)
        pts_j_valid = pts_j_valid[idx]
    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(pts_i_valid)
    pcd_j = o3d.geometry.PointCloud()
    pcd_j.points = o3d.utility.Vector3dVector(pts_j_valid)
    pcd_i = pcd_i.voxel_down_sample(voxel_size)
    pcd_j = pcd_j.voxel_down_sample(voxel_size)
    center_i = pcd_i.get_center()
    center_j = pcd_j.get_center()
    init_translate = center_i - center_j
    initial_transform = np.eye(4)
    initial_transform[:3, 3] = init_translate
    logger.info("[INFO] Running ICP with initial translation: [%.2f, %.2f, %.2f]", *init_translate)
    result = o3d.pipelines.registration.registration_icp(
        pcd_j, pcd_i,
        max_correspondence_distance=voxel_size * 5,
        init=initial_transform,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    logger.info("[INFO] ICP result: fitness=%.4f, rmse=%.4f", result.fitness, result.inlier_rmse)
    if result.fitness < 0.1:
        logger.warning("[WARN] ICP fitness too low (%.4f), skipping", result.fitness)
        return None
    T = result.transformation
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    U, _, Vt = np.linalg.svd(R)
    R_clean = (U @ Vt).astype(np.float32)
    if np.linalg.det(R_clean) < 0:
        Vt[2, :] *= -1
        R_clean = (U @ Vt).astype(np.float32)
    scale_i = np.std(pts_i_valid - center_i)
    scale_j = np.std(pts_j_valid - center_j)
    s = float(scale_i / max(scale_j, 1e-6))
    return s, R_clean, t, float(result.inlier_rmse)


def build_spanning_tree(block_pairs, num_blocks):
    parent = list(range(num_blocks))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True
    edges = []
    for bp in block_pairs:
        edges.append((bp["block_i"], bp["block_j"], bp.get("transform_err", 999.0), bp))
    edges.sort(key=lambda x: x[2])
    tree_edges = []
    for i, j, err, bp in edges:
        if union(i, j):
            tree_edges.append(bp)
            if len(tree_edges) == num_blocks - 1:
                break
    if len(tree_edges) < num_blocks - 1:
        logger.warning(
            "[WARN] Incomplete spanning tree: %d edges for %d blocks",
            len(tree_edges), num_blocks,
        )
    return tree_edges


def global_merge_blocks(blocks, block_data_list, config, output_path, use_icp_fallback=True):
    num_blocks = len(blocks)
    if num_blocks == 0:
        logger.error("[ERROR] No blocks to merge")
        return
    if num_blocks == 1:
        logger.info("[INFO] Only one block, copying to output")
        src = os.path.join(blocks[0]["output_dir"], "reconstruction_merged.ply")
        if os.path.isfile(src):
            import shutil
            os.makedirs(output_path, exist_ok=True)
            shutil.copy2(src, os.path.join(output_path, "reconstruction_global.ply"))
        return

    logger.info("[INFO] Step 1/4: Extracting cross-block image descriptors with SALAD...")
    detectors_info = []
    for idx, blk in enumerate(blocks):
        logger.info("[INFO]   Extracting descriptors for block %d: %s", idx, blk["name"])
        desc, img_paths = extract_block_descriptors(blk, config, sample_stride=10)
        detectors_info.append({"descriptors": desc, "image_paths": img_paths})

    logger.info("[INFO] Step 2/4: Finding cross-block image matches...")
    block_pairs = find_cross_block_matches(
        detectors_info, similarity_threshold=0.80, min_match_count=3
    )
    if len(block_pairs) == 0:
        logger.warning("[WARN] No cross-block matches found via SALAD!")
        if use_icp_fallback:
            logger.info("[INFO] Falling back to ICP-based alignment between adjacent blocks...")
            for i in range(num_blocks - 1):
                block_pairs.append({
                    "block_i": i, "block_j": i + 1,
                    "matches": [], "avg_similarity": 0.0, "num_matches": 0,
                })
        else:
            logger.error("[ERROR] Cannot align blocks without matches. Exiting.")
            return

    logger.info("[INFO] Step 3/4: Computing block-to-block transforms...")
    for bp in block_pairs:
        i, j = bp["block_i"], bp["block_j"]
        logger.info("[INFO]   Aligning block_%d <-> block_%d ...", i, j)
        result = compute_block_transform_via_points(
            block_data_list[i], block_data_list[j], config
        )
        if result is None and use_icp_fallback:
            logger.info("[INFO]   Point-based alignment failed, trying ICP...")
            result = compute_block_transform_via_icp(
                block_data_list[i], block_data_list[j], config
            )
        if result is not None:
            s, R, t, err = result
            bp["transform"] = (s, R, t)
            bp["transform_err"] = err
            logger.info("[INFO]   Block_%d -> Block_%d: s=%.6f, err=%.4f", i, j, s, err)
        else:
            bp["transform"] = None
            bp["transform_err"] = 999.0
            logger.warning("[WARN]   Failed to align block_%d <-> block_%d", i, j)

    valid_pairs = [bp for bp in block_pairs if bp.get("transform") is not None]
    if len(valid_pairs) == 0:
        logger.error("[ERROR] No valid block transforms computed. Cannot merge.")
        return

    tree_edges = build_spanning_tree(valid_pairs, num_blocks)
    adj = {i: [] for i in range(num_blocks)}
    for bp in tree_edges:
        i, j = bp["block_i"], bp["block_j"]
        s, R, t = bp["transform"]
        adj[i].append((j, s, R, t))
        s_inv = 1.0 / s
        R_inv = R.T
        t_inv = -s_inv * (R_inv @ t)
        adj[j].append((i, s_inv, R_inv, t_inv))

    root = 0
    block_transforms = {root: (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
    visited = set()
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        s_node, R_node, t_node = block_transforms[node]
        for neighbor, s_rel, R_rel, t_rel in adj[node]:
            if neighbor in visited:
                continue
            s_new = s_node * s_rel
            R_new = R_node @ R_rel
            t_new = s_node * (R_node @ t_rel) + t_node
            block_transforms[neighbor] = (s_new, R_new, t_new)
            queue.append(neighbor)

    for i in range(num_blocks):
        if i not in block_transforms:
            logger.warning("[WARN] Block %d disconnected from root, keeping local coords", i)
            block_transforms[i] = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

    logger.info("[INFO] Step 4/4: Applying global transforms and merging...")
    global_pcd_dir = os.path.join(output_path, "tmp_global_pcd")
    os.makedirs(global_pcd_dir, exist_ok=True)

    for idx, blk in enumerate(blocks):
        s, R, t = block_transforms[idx]
        logger.info(
            "[INFO]   Block %d (%s): s=%.6f, |t|=%.4f",
            idx, blk["name"], s, float(np.linalg.norm(t)),
        )
        data = block_data_list[idx]
        if data is None:
            logger.warning("[WARN]   No data for block %d, skipping", idx)
            continue
        world_points = data["world_points"]
        confs = data["conf"]
        colors = data["images"]
        world_points = apply_sim3_direct_torch(world_points, s, R, t)
        conf_threshold = np.mean(confs) * config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
        ply_path = os.path.join(global_pcd_dir, "block_{}_pcd.ply".format(idx))
        save_confident_pointcloud_batch(
            points=world_points, colors=colors, confs=confs,
            output_path=ply_path, conf_threshold=conf_threshold,
            sample_ratio=config["Model"]["Pointcloud_Save"]["sample_ratio"],
        )
        aligned_data = {"world_points": world_points, "conf": confs, "images": colors}
        np.save(
            os.path.join(output_path, "block_{}_global_aligned.npy".format(idx)),
            aligned_data, allow_pickle=True,
        )

    merged_path = os.path.join(output_path, "reconstruction_global.ply")
    merge_ply_files(global_pcd_dir, merged_path)
    logger.info("[INFO] Global merge complete: %s", merged_path)

    transform_path = os.path.join(output_path, "block_transforms.npy")
    serializable = {}
    for k, (s, R, t) in block_transforms.items():
        serializable[str(k)] = {"s": float(s), "R": R.tolist(), "t": t.tolist()}
    np.save(transform_path, serializable, allow_pickle=True)
    logger.info("[INFO] Block transforms saved: %s", transform_path)


def main():
    parser = argparse.ArgumentParser(description="Cross-block global alignment and merging")
    parser.add_argument("--base_output_dir", type=str,
                        default=str(_REPO_ROOT / "exp" / "matrixcity"),
                        help="Base directory containing per-block output folders")
    parser.add_argument("--city_size", type=str, default="big_city",
                        help="small_city or big_city")
    parser.add_argument("--split", type=str, default="train",
                        help="train or test")
    parser.add_argument("--model_name", type=str, default="depthanything3",
                        help="Model name used for per-block inference")
    parser.add_argument("--output_path", type=str, default="",
                        help="Output directory for global merge results")
    parser.add_argument("--config_path", type=str,
                        default=str(_REPO_ROOT / "configs" / "base_config.yaml"),
                        help="Base config path")
    parser.add_argument("--run_args_yaml", type=str, default="",
                        help="YAML config with GlobalMerge section (overrides defaults)")
    parser.add_argument("--use_icp_fallback", type=int, default=-1,
                        help="Use ICP as fallback when SALAD matching fails (-1=from config)")
    parser.add_argument("--max_chunks_per_block", type=int, default=-1,
                        help="Max chunks to load per block (-1=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config_path)

    if args.run_args_yaml and os.path.isfile(args.run_args_yaml):
        run_config = load_config(args.run_args_yaml)
        gm = run_config.get("GlobalMerge", {})
        if args.base_output_dir == str(_REPO_ROOT / "exp" / "matrixcity") and gm.get("base_output_dir"):
            args.base_output_dir = gm["base_output_dir"]
        if args.city_size == "big_city" and gm.get("city_size"):
            args.city_size = gm["city_size"]
        if args.split == "train" and gm.get("split"):
            args.split = gm["split"]
        if args.model_name == "depthanything3" and gm.get("model_name"):
            args.model_name = gm["model_name"]
        if not args.output_path and gm.get("output_path"):
            args.output_path = gm["output_path"]
        if gm.get("config_path"):
            config = load_config(gm["config_path"])
        if args.use_icp_fallback == -1 and "use_icp_fallback" in gm:
            args.use_icp_fallback = int(gm["use_icp_fallback"])
        if args.max_chunks_per_block == -1 and "max_chunks_per_block" in gm:
            args.max_chunks_per_block = int(gm["max_chunks_per_block"])
        icp_cfg = gm.get("ICP", {})
        for k, v in icp_cfg.items():
            config.setdefault("GlobalMerge", {}).setdefault("ICP", {})[k] = v
        salad_cfg = gm.get("SALAD", {})
        for k, v in salad_cfg.items():
            config.setdefault("GlobalMerge", {}).setdefault("SALAD", {})[k] = v
        pc_cfg = gm.get("Pointcloud_Save", {})
        for k, v in pc_cfg.items():
            config["Model"]["Pointcloud_Save"][k] = v
        logger.info("[INFO] Loaded run config from %s", args.run_args_yaml)

    if args.use_icp_fallback == -1:
        args.use_icp_fallback = 1

    if not args.output_path:
        args.output_path = os.path.join(
            args.base_output_dir,
            "global_merge_{}_{}_{}".format(args.model_name, args.city_size, args.split),
        )

    logger.info("[INFO] Discovering block outputs...")
    blocks = discover_block_outputs(
        args.base_output_dir, args.city_size, args.split, args.model_name
    )
    if len(blocks) == 0:
        logger.error("[ERROR] No block outputs found")
        return

    logger.info("[INFO] Found %d blocks: %s", len(blocks), [b["name"] for b in blocks])

    logger.info("[INFO] Loading block point data...")
    block_data_list = []
    for blk in blocks:
        data = load_block_point_data(blk, max_chunks=args.max_chunks_per_block)
        block_data_list.append(data)
        if data is not None:
            logger.info("[INFO]   Block %s: points shape=%s", blk["name"], data["world_points"].shape)
        else:
            logger.warning("[WARN]   Block %s: no data loaded", blk["name"])

    os.makedirs(args.output_path, exist_ok=True)
    global_merge_blocks(
        blocks, block_data_list, config, args.output_path,
        use_icp_fallback=bool(args.use_icp_fallback),
    )


if __name__ == "__main__":
    main()
