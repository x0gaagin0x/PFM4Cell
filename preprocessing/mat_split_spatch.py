import argparse
import os
import glob
import math
import numpy as np
import torch
from typing import Dict, List, Tuple
from PIL import Image
from scipy.io import loadmat
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import subgraph

# ==============================
# Configuration
# ==============================
IMG_DIR = None
MAT_DIR = None
PATCH_IMG_OUT = None
PATCH_GRAPH_OUT = None

PATCH_SIZE = 224
K_NEIGHBORS = 12
RESIZE_SCALE = 1.0

# ==============================
# Utilities
# ==============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)

# ==============================
# Loading and annotation parsing
# ==============================
def load_image(path: str) -> Image.Image:
    # Use robust loading for large images if necessary
    Image.MAX_IMAGE_PIXELS = None 
    img = Image.open(path).convert("RGB")
    return img

def load_gt_mat(path: str) -> Dict[str, np.ndarray]:
    """
    Expected keys:
      - 'id'      : shape (S, 1) -> (S,), int
      - 'class'   : shape (S, 1) -> (S,), uint8/int
      - 'bbox'    : shape (S, 4), [ymin, ymax, xmin, xmax]
      - 'centroid': shape (S, 2), [x, y]
      - 'expr'    : shape (S, F), features
    """
    mat = loadmat(path)

    # Helper to squeeze dimensions safely
    def squeeze(a):
        a = np.array(a)
        return np.squeeze(a)

    # 1. Load basic info to determine number of instances (S)
    ids = squeeze(mat['id']).astype(np.int32)
    # Handle single instance case where squeeze might result in scalar
    if ids.ndim == 0:
        ids = ids[None]
    
    S = ids.shape[0]

    classes = squeeze(mat['class']).astype(np.int32)
    if classes.ndim == 0: classes = classes[None]

    bboxes = squeeze(mat['bbox']).astype(np.int32)
    if bboxes.ndim == 1: bboxes = bboxes[None, :]

    centroids = squeeze(mat['centroid']).astype(np.float32)
    if centroids.ndim == 1: centroids = centroids[None, :]

    # --- UPDATED: Load 'expr' ---
    # We use reshape(S, -1) to ensure it is always (S, Features), 
    # even if there is only 1 instance or 1 feature.
    expr = np.array(mat['expr'], dtype=np.float32)
    expr = expr.reshape(S, -1)

    # Sanity checks
    assert classes.shape[0] == S, f"Class count mismatch: {classes.shape[0]} vs {S}"
    assert bboxes.shape[0] == S, f"BBox count mismatch: {bboxes.shape[0]} vs {S}"
    assert centroids.shape[0] == S, f"Centroid count mismatch: {centroids.shape[0]} vs {S}"
    assert expr.shape[0] == S, f"Expr count mismatch: {expr.shape[0]} vs {S}"

    return {
        "id": ids, 
        "class": classes, 
        "bbox": bboxes, 
        "centroid": centroids,
        "expr": expr  # <--- Added
    }

# ==============================
# Geometry transforms
# ==============================
def resize_image_and_ann(img: Image.Image,
                         ann: Dict[str, np.ndarray],
                         scale: float = RESIZE_SCALE) -> Tuple[Image.Image, Dict[str, np.ndarray]]:
    """
    Resize image by 'scale' and multiply spatial coordinates.
    'expr' features are copied as-is.
    """
    W, H = img.size
    
    # If scale is 1.0, skip actual resize operation to save time/quality
    if scale != 1.0:
        new_W, new_H = int(round(W * scale)), int(round(H * scale))
        img_resized = img.resize((new_W, new_H), resample=Image.BICUBIC)
    else:
        img_resized = img

    bboxes = ann["bbox"].copy()
    centroids = ann["centroid"].copy()

    # bbox format: [ymin, ymax, xmin, xmax]
    bboxes[:, 0:2] = (bboxes[:, 0:2] * scale).astype(np.int32)
    bboxes[:, 2:4] = (bboxes[:, 2:4] * scale).astype(np.int32)

    # centroid format: [x, y]
    centroids = (centroids * scale).astype(np.float32)

    ann_scaled = {
        "id": ann["id"].copy(),
        "class": ann["class"].copy(),
        "bbox": bboxes,
        "centroid": centroids,
        "expr": ann["expr"].copy()  # <--- Keep expr
    }
    return img_resized, ann_scaled

def pad_right_bottom_to_multiple(img: Image.Image,
                                 multiple: int = PATCH_SIZE,
                                 pad_color: int = 255) -> Tuple[Image.Image, Tuple[int, int]]:
    W, H = img.size
    target_W = int(math.ceil(W / multiple) * multiple)
    target_H = int(math.ceil(H / multiple) * multiple)

    pad_right = target_W - W
    pad_bottom = target_H - H

    if pad_right == 0 and pad_bottom == 0:
        return img, (0, 0)

    canvas = Image.new("RGB", (target_W, target_H), color=(pad_color, pad_color, pad_color))
    canvas.paste(img, (0, 0))
    return canvas, (pad_right, pad_bottom)

# ==============================
# Patching
# ==============================
def iter_patch_windows(W: int, H: int, patch: int = PATCH_SIZE) -> List[Tuple[int, int, int, int]]:
    windows = []
    for y0 in range(0, H, patch):
        for x0 in range(0, W, patch):
            x1 = x0 + patch
            y1 = y0 + patch
            windows.append((x0, y0, x1, y1))
    return windows

def patch_center(x0: int, y0: int, patch: int = PATCH_SIZE) -> Tuple[int, int]:
    cx = x0 + patch // 2
    cy = y0 + patch // 2
    return cx, cy

def save_patch(img_np: np.ndarray, x0: int, y0: int, patch: int,
               base_name: str, out_dir: str):
    cx, cy = patch_center(x0, y0, patch)
    patch_arr = img_np[y0:y0+patch, x0:x0+patch, :]
    out_name = f"{base_name}_{cx}x{cy}.png"
    ensure_dir(out_dir)
    Image.fromarray(patch_arr).save(os.path.join(out_dir, out_name))

def split_and_save_patches(img: Image.Image, base_name: str, out_dir: str, patch: int = PATCH_SIZE) -> List[Dict]:
    W, H = img.size
    img_np = pil_to_np(img)
    windows = iter_patch_windows(W, H, patch)
    meta = []
    for x0, y0, x1, y1 in windows:
        cx, cy = patch_center(x0, y0, patch)
        name = f"{base_name}_{cx}x{cy}.png"
        save_patch(img_np, x0, y0, patch, base_name, out_dir)
        meta.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "center": (cx, cy), "name": name})
    return meta

# ==============================
# Graph construction
# ==============================
def build_global_graph(ann: Dict[str, np.ndarray], k: int = K_NEIGHBORS) -> Data:
    """
    Build KNN graph. 
    Added 'x_expr' to node attributes.
    """
    centroids = ann["centroid"]
    bboxes = ann["bbox"]
    classes = ann["class"]
    ids = ann["id"]
    expr = ann["expr"]  # <--- Get expr

    # 1-based class to 0-based
    classes_0 = (classes - 1).astype(np.int64)

    # Create Tensors
    pos = torch.from_numpy(centroids).float()  # [S, 2]
    x_bbox = torch.from_numpy(bboxes).int()
    x_centroid = pos.clone()
    y_class = torch.from_numpy(classes_0).long().view(-1)
    node_id = torch.from_numpy(ids).long().view(-1)
    x_expr = torch.from_numpy(expr).float()    # <--- Tensorize expr

    # Build Edge Index (KNN)
    edge_index = knn_graph(pos, k=k, loop=False)

    data = Data(
        edge_index=edge_index,
        x_bbox=x_bbox,
        x_centroid=x_centroid,
        x_expr=x_expr,         # <--- Store in Data
        y_class=y_class,
        node_id=node_id,
    )
    return data

# ==============================
# Subgraph extraction per patch
# ==============================
def nodes_in_patch(centroids: torch.Tensor,
                   x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    xs = centroids[:, 0]
    ys = centroids[:, 1]
    mask = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return torch.nonzero(mask, as_tuple=False).view(-1)

def extract_patch_subgraph(global_data: Data,
                           window: Dict) -> Data:
    x0, y0, x1, y1 = window["x0"], window["y0"], window["x1"], window["y1"]
    cx, cy = window["center"]

    node_idx = nodes_in_patch(global_data.x_centroid, x0, y0, x1, y1)

    if node_idx.numel() == 0:
        sub = Data(num_nodes=0, edge_index=torch.empty(2, 0, dtype=torch.long))
        sub.patch_center = torch.tensor([[cx, cy]], dtype=torch.int32)
        sub.window = torch.tensor([x0, y0, x1, y1], dtype=torch.int32)
        return sub

    # Extract subgraph topology
    eidx, _ = subgraph(node_idx, global_data.edge_index, relabel_nodes=True)

    # Create new Data object with sliced attributes
    sub = Data(
        num_nodes=global_data.x_bbox[node_idx].shape[0],
        edge_index=eidx,
        x_bbox=global_data.x_bbox[node_idx],
        x_centroid=global_data.x_centroid[node_idx],
        x_expr=global_data.x_expr[node_idx],     # <--- Slice expr
        y_class=global_data.y_class[node_idx],
        node_id=global_data.node_id[node_idx],
    )
    sub.patch_center = torch.tensor([[cx, cy]], dtype=torch.int32)
    sub.window = torch.tensor([x0, y0, x1, y1], dtype=torch.int32)
    return sub

def save_subgraph(sub: Data, base_name: str, center: Tuple[int, int], out_dir: str):
    cx, cy = center
    out_name = f"{base_name}_{cx}x{cy}.pt"
    ensure_dir(out_dir)
    torch.save(sub, os.path.join(out_dir, out_name))

# ==============================
# End-to-end processing for one pair
# ==============================
def process_one(image_path: str, mat_path: str):
    base = stem(image_path)

    # 1) Load and resize
    img = load_image(image_path)
    ann = load_gt_mat(mat_path)
    img_resized, ann_resized = resize_image_and_ann(img, ann, scale=RESIZE_SCALE)

    # 2) Pad right/bottom
    img_padded, _ = pad_right_bottom_to_multiple(img_resized, multiple=PATCH_SIZE, pad_color=255)

    # 3) Split and save patches
    patch_meta = split_and_save_patches(img_padded, base, PATCH_IMG_OUT, patch=PATCH_SIZE)

    # 4) Global graph (now includes x_expr)
    global_graph = build_global_graph(ann_resized, k=K_NEIGHBORS)

    # 5) Per-patch subgraphs
    for meta in patch_meta:
        sub = extract_patch_subgraph(global_graph, meta)
        if sub.num_nodes > 0:
            save_subgraph(sub, base, meta["center"], PATCH_GRAPH_OUT)

# ==============================
# Batch processing over folders
# ==============================
def find_pairs(img_dir: str, mat_dir: str) -> List[Tuple[str, str]]:
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mats_by_stem = {}
    for mp in glob.glob(os.path.join(mat_dir, "*.mat")):
        mats_by_stem[stem(mp)] = mp

    pairs = []
    for ip in img_paths:
        s = stem(ip)
        if s in mats_by_stem:
            pairs.append((ip, mats_by_stem[s]))
        else:
            # Heuristic match
            parts = s.split("_")
            if len(parts) > 1:
                s2 = "_".join(parts[:-1])
                if s2 in mats_by_stem:
                    pairs.append((ip, mats_by_stem[s2]))
                else:
                    s3 = parts[0]
                    if s3 in mats_by_stem:
                        pairs.append((ip, mats_by_stem[s3]))
    return pairs

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split padded WSIs into 224x224 patches and per-patch KNN subgraphs."
    )
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Dataset directory. Defaults read images/ and labels/, and write patches and graphs beside them")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="PNG directory (default: <base_dir>/images)")
    parser.add_argument("--mat_dir", type=str, default=None,
                        help="MAT directory (default: <base_dir>/labels)")
    parser.add_argument("--patch_out", type=str, default=None,
                        help="Patch image output (default: <base_dir>/patches4UNI)")
    parser.add_argument("--graph_out", type=str, default=None,
                        help="Graph output (default: <base_dir>/graphs_manual_anno4UNI)")
    return parser.parse_args()


def main():
    global IMG_DIR, MAT_DIR, PATCH_IMG_OUT, PATCH_GRAPH_OUT
    args = parse_args()
    IMG_DIR = args.img_dir or os.path.join(args.base_dir, "images")
    MAT_DIR = args.mat_dir or os.path.join(args.base_dir, "labels")
    PATCH_IMG_OUT = args.patch_out or os.path.join(args.base_dir, "patches4UNI")
    PATCH_GRAPH_OUT = args.graph_out or os.path.join(args.base_dir, "graphs_manual_anno4UNI")

    ensure_dir(PATCH_IMG_OUT)
    ensure_dir(PATCH_GRAPH_OUT)

    pairs = find_pairs(IMG_DIR, MAT_DIR)
    if not pairs:
        print("No image/mat pairs found; check directories and naming.")
        return

    for img_path, mat_path in pairs:
        print(f"Processing: {img_path}  |  {mat_path}")
        process_one(img_path, mat_path)

    print("Done.")

if __name__ == "__main__":
    main()