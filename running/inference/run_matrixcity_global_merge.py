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


def _load_ply_as_array(ply_path):
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        if colors.ndim == 2 and colors.shape[1] == 3:
            colors = (colors * 255.0).clip(0, 255).astype(np.uint8)
        else:
            colors = None
        return pts, colors
    except ImportError:
        pass
    try:
        import trimesh
        mesh = trimesh.load(ply_path)
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        colors = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
            vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)
            if vc.ndim == 2 and vc.shape[1] >= 3:
                colors = (vc[:, :3] * 255.0).clip(0, 255).astype(np.uint8)
        return pts, colors
    except ImportError:
        pass
    pts_list = []
    colors_list = []
    with open(ply_path, "rb") as f:
        header = b""
        num_verts = 0
        has_color = False
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
            if line.startswith(b"element vertex"):
                num_verts = int(line.split()[2])
            if b"red" in line.lower() or b"green" in line.lower():
                has_color = True
        for _ in range(num_verts):
            vals = f.readline().decode("ascii", errors="replace").strip().split()
            if len(vals) >= 3:
                pts_list.append([float(vals[0]), float(vals[1]), float(vals[2])])
                if has_color and len(vals) >= 6:
                    colors_list.append([int(vals[3]), int(vals[4]), int(vals[5])])
    pts = np.array(pts_list, dtype=np.float32)
    colors = np.array(colors_list, dtype=np.uint8) if colors_list else None
    return pts, colors


def load_block_point_data(block_info, max_chunks=None):
    aligned_dir = os.path.join(block_info["output_dir"], "tmp_predictions_aligned")
    pcd_dir = os.path.join(block_info["output_dir"], "tmp_predictions_pcd")

    if os.path.isdir(aligned_dir):
        chunk_files = sorted(glob.glob(os.path.join(aligned_dir, "chunk_*.npy")))
        if len(chunk_files) > 0:
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
            if len(all_points) > 0:
                return {
                    "world_points": np.concatenate(all_points, axis=0),
                    "conf": np.concatenate(all_conf, axis=0),
                    "images": np.concatenate(all_colors, axis=0) if all_colors else None,
                    "source": "npy",
                }

    if os.path.isdir(pcd_dir):
        ply_files = sorted(glob.glob(os.path.join(pcd_dir, "*_pcd.ply")))
        if len(ply_files) > 0:
            logger.info("[INFO] Loading block %s from PLY files (%d files)", block_info["name"], len(ply_files))
            all_pts = []
            all_colors = []
            for pf in ply_files:
                try:
                    pts, colors = _load_ply_as_array(pf)
                    if pts is not None and len(pts) > 0:
                        all_pts.append(pts)
                        if colors is not None:
                            all_colors.append(colors)
                except Exception as e:
                    logger.warning("[WARN] Failed to load PLY %s: %s", pf, str(e))
            if len(all_pts) > 0:
                pts = np.concatenate(all_pts, axis=0)
                colors = np.concatenate(all_colors, axis=0) if all_colors else None
                conf = np.ones(pts.shape[:-1] if pts.ndim > 1 else (pts.shape[0],), dtype=np.float32)
                return {"world_points": pts, "conf": conf, "images": colors, "source": "ply"}

    logger.warning("[WARN] No point data found for block %s", block_info["name"])
    return None


def extract_block_descriptors(block_info, config, sample_stride=10, dataset_path=None, city_size=None, split=None):
    from PIL import Image as PILImage
    image_cache_dir = os.path.join(block_info["output_dir"], "block_images_for_vpr")
    if not os.path.isdir(image_cache_dir) or not os.listdir(image_cache_dir):
        os.makedirs(image_cache_dir, exist_ok=True)
        extracted = False
        unaligned_dir = os.path.join(block_info["output_dir"], "tmp_predictions_unaligned")
        if os.path.isdir(unaligned_dir):
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
                    extracted = True
                except Exception as e:
                    logger.warning("[WARN] Failed to extract images from %s: %s", cf, str(e))
        if not extracted and dataset_path and city_size and split:
            block_name = block_info["name"]
            area_path = os.path.join(dataset_path, city_size, "aerial", split, block_name)
            if os.path.isdir(area_path):
                cam_dirs = sorted([
                    d for d in os.listdir(area_path)
                    if os.path.isdir(os.path.join(area_path, d)) and d.isdigit()
                ])
                count = 0
                if len(cam_dirs) > 0:
                    for cam_dir in cam_dirs:
                        cam_path = os.path.join(area_path, cam_dir)
                        img_files = sorted(glob.glob(os.path.join(cam_path, "*.png")) + glob.glob(os.path.join(cam_path, "*.jpg")))
                        for i in range(0, len(img_files), sample_stride):
                            try:
                                img = PILImage.open(img_files[i]).convert("RGB")
                                img = img.resize((224, 224))
                                img.save(os.path.join(image_cache_dir, "frame_{:06d}.png".format(count)))
                                count += 1
                            except Exception as e:
                                logger.warning("[WARN] Failed to load image %s: %s", img_files[i], str(e))
                        if count > 0:
                            break
                else:
                    img_files = sorted(glob.glob(os.path.join(area_path, "*.png")) + glob.glob(os.path.join(area_path, "*.jpg")))
                    for i in range(0, len(img_files), sample_stride):
                        try:
                            img = PILImage.open(img_files[i]).convert("RGB")
                            img = img.resize((224, 224))
                            img.save(os.path.join(image_cache_dir, "frame_{:06d}.png".format(count)))
                            count += 1
                        except Exception as e:
                            logger.warning("[WARN] Failed to load image %s: %s", img_files[i], str(e))
    if not os.path.isdir(image_cache_dir) or not os.listdir(image_cache_dir):
        logger.warning("[WARN] No images extracted for VPR in block %s", block_info["name"])
        return None, []
    detector = LoopDetector(image_cache_dir, config=config)
    try:
        detector.load_model()
    except Exception as e:
        logger.warning("[WARN] Failed to load SALAD model (network or cache issue): %s", str(e))
        logger.warning("[WARN] Falling back to ICP-only alignment for this block")
        return None, []
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
    if point_map_i.ndim == 2:
        point_map_i = point_map_i[np.newaxis, ..., np.newaxis, :]
        conf_map_i = conf_map_i[np.newaxis, ..., np.newaxis]
    if point_map_j.ndim == 2:
        point_map_j = point_map_j[np.newaxis, ..., np.newaxis, :]
        conf_map_j = conf_map_j[np.newaxis, ..., np.newaxis]
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
    conf_thresh_i = np.mean(conf_i) * 0.5
    conf_thresh_j = np.mean(conf_j) * 0.5
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
    center_i = np.asarray(pcd_i.points).mean(axis=0)
    center_j = np.asarray(pcd_j.points).mean(axis=0)
    scale_i = float(np.std(np.asarray(pcd_i.points) - center_i))
    scale_j = float(np.std(np.asarray(pcd_j.points) - center_j))

    best_result = None
    best_fitness = -1.0

    init_transforms = []
    init_translate = center_i - center_j
    T1 = np.eye(4)
    T1[:3, 3] = init_translate
    init_transforms.append(T1)

    s_est = scale_i / max(scale_j, 1e-6)
    T2 = np.eye(4)
    T2[:3, :3] *= s_est
    T2[:3, 3] = init_translate
    init_transforms.append(T2)

    for rot_angle in [0, 90, 180, 270]:
        angle_rad = np.radians(rot_angle)
        for axis_idx in range(3):
            axis = np.zeros(3)
            axis[axis_idx] = 1.0
            c, s_val = np.cos(angle_rad), np.sin(angle_rad)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R_rot = np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s_val * K
            T_try = np.eye(4)
            T_try[:3, :3] = R_rot * s_est
            T_try[:3, 3] = init_translate
            init_transforms.append(T_try)

    for init_T in init_transforms:
        try:
            result = o3d.pipelines.registration.registration_icp(
                pcd_j, pcd_i,
                max_correspondence_distance=voxel_size * 10,
                init=init_T,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_result = result
        except Exception:
            continue

    if best_result is None or best_result.fitness < 0.05:
        logger.warning("[WARN] ICP fitness too low, skipping")
        return None

    logger.info("[INFO] ICP result: fitness=%.4f, rmse=%.4f", best_result.fitness, best_result.inlier_rmse)

    T = best_result.transformation
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    U, _, Vt = np.linalg.svd(R)
    R_clean = (U @ Vt).astype(np.float32)
    if np.linalg.det(R_clean) < 0:
        Vt[2, :] *= -1
        R_clean = (U @ Vt).astype(np.float32)
    s = float(scale_i / max(scale_j, 1e-6))
    return s, R_clean, t, float(best_result.inlier_rmse)


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


def global_merge_blocks(blocks, block_data_list, config, output_path, use_icp_fallback=True, dataset_path=None, city_size=None, split=None):
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
        desc, img_paths = extract_block_descriptors(blk, config, sample_stride=10, dataset_path=dataset_path, city_size=city_size, split=split)
        detectors_info.append({"descriptors": desc, "image_paths": img_paths})

    logger.info("[INFO] Step 2/4: Finding cross-block image matches...")
    block_pairs = find_cross_block_matches(
        detectors_info, similarity_threshold=0.80, min_match_count=3
    )
    if len(block_pairs) == 0:
        logger.warning("[WARN] No cross-block matches found via SALAD!")
        if use_icp_fallback:
            logger.info("[INFO] Falling back to ICP-based alignment for all block pairs...")
            for i in range(num_blocks):
                for j in range(i + 1, num_blocks):
                    block_pairs.append({
                        "block_i": i, "block_j": j,
                        "matches": [], "avg_similarity": 0.0, "num_matches": 0,
                    })
        else:
            logger.error("[ERROR] Cannot align blocks without matches. Exiting.")
            return

    logger.info("[INFO] Step 3/4: Computing block-to-block transforms...")
    for bp in block_pairs:
        i, j = bp["block_i"], bp["block_j"]
        logger.info("[INFO]   Aligning block_%d <-> block_%d ...", i, j)
        src_i = block_data_list[i].get("source", "npy") if block_data_list[i] else "npy"
        src_j = block_data_list[j].get("source", "npy") if block_data_list[j] else "npy"
        result = None
        if src_i == "npy" and src_j == "npy":
            result = compute_block_transform_via_points(
                block_data_list[i], block_data_list[j], config
            )
        if result is None and use_icp_fallback:
            logger.info("[INFO]   Trying ICP alignment...")
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
        needs_reshape = world_points.ndim == 2
        if needs_reshape:
            world_points = world_points[np.newaxis, ..., np.newaxis, :]
        world_points = apply_sim3_direct_torch(world_points, s, R, t)
        if needs_reshape:
            world_points = world_points.reshape(-1, 3)
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
    parser.add_argument("--use_icp_fallback", type=int, default=1,
                        help="Use ICP as fallback when SALAD matching fails")
    parser.add_argument("--max_chunks_per_block", type=int, default=-1,
                        help="Max chunks to load per block (-1=all)")
    parser.add_argument("--dataset_path", type=str,
                        default=str(_REPO_ROOT / "dataset" / "MatrixCity"),
                        help="Root of MatrixCity dataset for VPR image extraction")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config_path)

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
        dataset_path=args.dataset_path, city_size=args.city_size, split=args.split,
    )


if __name__ == "__main__":
    main()
