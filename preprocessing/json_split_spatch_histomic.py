import argparse
import os
import math
import glob
import json
import copy
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import cv2
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import subgraph
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
import skimage.measure

# 1. Disable the safety limit for massive images in Pillow
Image.MAX_IMAGE_PIXELS = None 

# ==============================
# Configuration
# ==============================
BASE_DIR = None
IMG_DIR = None
JSON_DIR = None
GRAPH_OUT_DIR = None

PATCH_SIZE = 224
K_NEIGHBORS = 12
RESIZE_SCALE = 1.0  
DELAUNAY_DIST_THRESH = 200

# ==============================
# Import HistomicsTK Scripts (Optional)
# ==============================
try:
    from histomicstk_scripts import compute_fsd_features, compute_intensity_features, compute_gradient_features
    HAS_HISTOMICSTK = True
    print("Successfully imported HistomicsTK scripts.")
except ImportError:
    warnings.warn("HistomicsTK scripts not found. Related features will be 0.")
    HAS_HISTOMICSTK = False

# ==============================
# Feature Extraction Classes (From Script 2)
# ==============================
class NucleiFeatureExtractor:
    def __init__(self): pass

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return np.nan_to_num(gray, nan=0.0).astype(np.uint8)

    def get_feature_names_base(self):
        feat_color = [f'{c}_{s}' for c in ['Grey', 'R', 'G', 'B'] for s in ['mean', 'std', 'min', 'max']]
        feat_cyto = ['cyto_offset', 'cyto_area_of_bbox', 'cyto_bg_mask_sum', 'cyto_bg_mask_ratio', 
                     'cyto_cytomask_sum', 'cyto_cytomask_ratio']
        feat_cyto += [f'cyto_{c}_{s}' for c in ['Grey', 'R', 'G', 'B'] for s in ['mean', 'std', 'min', 'max']]
        feat_morph = ['major_axis_length', 'minor_axis_length', 'major_minor_ratio', 
                      'orientation', 'orientation_degree', 'area', 'extent', 'solidity', 
                      'convex_area', 'Eccentricity', 'equivalent_diameter', 'perimeter', 'perimeter_crofton']
        feat_haralick = ['contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation', 'heterogeneity']
        feat_gradient = ['Gradient.Mag.Mean', 'Gradient.Mag.Std', 'Gradient.Mag.Skewness', 'Gradient.Mag.Kurtosis', 
                         'Gradient.Mag.HistEntropy', 'Gradient.Mag.HistEnergy', 'Gradient.Canny.Sum', 'Gradient.Canny.Mean']
        feat_intensity = ['Intensity.Min', 'Intensity.Max', 'Intensity.Mean', 'Intensity.Median', 
                          'Intensity.MeanMedianDiff', 'Intensity.Std', 'Intensity.IQR', 'Intensity.MAD', 
                          'Intensity.Skewness', 'Intensity.Kurtosis', 'Intensity.HistEnergy', 'Intensity.HistEntropy']
        feat_fsd = [f'Shape.FSD{i+1}' for i in range(6)]
        return feat_color + feat_cyto + feat_morph + feat_haralick + feat_gradient + feat_intensity + feat_fsd

    def get_cytoplasm_features(self, nuclei_img_np, mask_binary, all_objects_mask, offset=20, dilation_kernel=5):
        mask_bool = mask_binary.astype(bool)
        all_objects_bool = all_objects_mask.astype(bool)
        mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8))
        mask_dilated = np.array(mask_pil.filter(ImageFilter.MaxFilter(dilation_kernel))) > 0
        cytoplasm_mask = mask_dilated & (~mask_bool) & (~all_objects_bool)
        
        cytoplasm_img_np = copy.deepcopy(nuclei_img_np).astype(float)
        cytoplasm_img_np[~cytoplasm_mask] = np.nan
        
        stat_cyto = {}
        h, w = nuclei_img_np.shape[:2]
        stat_cyto['cyto_offset'] = offset
        stat_cyto['cyto_area_of_bbox'] = h * w
        bg_mask = ~all_objects_bool
        stat_cyto['cyto_bg_mask_sum'] = np.sum(bg_mask)
        stat_cyto['cyto_bg_mask_ratio'] = stat_cyto['cyto_bg_mask_sum'] / (h * w + 1e-6)
        stat_cyto['cyto_cytomask_sum'] = np.sum(cytoplasm_mask)
        stat_cyto['cyto_cytomask_ratio'] = stat_cyto['cyto_cytomask_sum'] / (h * w + 1e-6)

        if np.nansum(cytoplasm_img_np) == 0:
            for k in ['Grey', 'R', 'G', 'B']:
                stat_cyto[f'cyto_{k}_mean'] = 255
                stat_cyto[f'cyto_{k}_std'] = 0
                stat_cyto[f'cyto_{k}_min'] = 255
                stat_cyto[f'cyto_{k}_max'] = 255
        else:
            cytoplasm_img_np_grey = self.rgb2gray(cytoplasm_img_np).astype(float)
            cytoplasm_img_np_grey[np.isnan(cytoplasm_img_np[...,0])] = np.nan
            stat_cyto['cyto_Grey_mean'] = np.nanmean(cytoplasm_img_np_grey)
            stat_cyto['cyto_Grey_std'] = np.nanstd(cytoplasm_img_np_grey)
            stat_cyto['cyto_Grey_min'] = np.nanmin(cytoplasm_img_np_grey)
            stat_cyto['cyto_Grey_max'] = np.nanmax(cytoplasm_img_np_grey)
            for i, col in enumerate(['R', 'G', 'B']):
                c_data = cytoplasm_img_np[:,:,i]
                stat_cyto[f'cyto_{col}_mean'] = np.nanmean(c_data)
                stat_cyto[f'cyto_{col}_std'] = np.nanstd(c_data)
                stat_cyto[f'cyto_{col}_min'] = np.nanmin(c_data)
                stat_cyto[f'cyto_{col}_max'] = np.nanmax(c_data)
        return stat_cyto

    def get_haralick_features(self, nuclei_img_object, area, quantization=10):
        nuclei_img_2 = copy.deepcopy(nuclei_img_object)
        nuclei_img_2[np.isnan(nuclei_img_2)] = 255
        nuclei_img_2 = nuclei_img_2.astype(np.uint8)
        magnification = 40 
        resolution = int(np.max([1, np.round(1 / int(magnification) * area * 0.002)]))
        level = np.int16(255/quantization)+1
        nuclei_img_2_gray = self.rgb2gray(nuclei_img_2/quantization)
        glcm = graycomatrix(nuclei_img_2_gray, distances=[resolution], 
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=level, symmetric=False, normed=True)
        glcm = glcm[0:level-1, 0:level-1, :, :]
        stat_haralick = {}
        for v in ['contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation']:
            stat_haralick[v] = np.mean(graycoprops(glcm, v))
        stat_haralick['heterogeneity'] = 1 - stat_haralick['homogeneity']
        return stat_haralick

    def process_single_nucleus(self, whole_img, inst_map, nucleus_id, bbox):
        ymin, ymax, xmin, xmax = bbox
        pad = 20
        h, w = whole_img.shape[:2]
        y1, y2 = max(0, ymin-pad), min(h, ymax+pad)
        x1, x2 = max(0, xmin-pad), min(w, xmax+pad)
        
        crop_img = whole_img[y1:y2, x1:x2]
        crop_inst = inst_map[y1:y2, x1:x2]
        
        mask_binary = (crop_inst == nucleus_id).astype(np.uint8)
        all_objects_mask = (crop_inst > 0).astype(np.uint8)
        
        props = skimage.measure.regionprops(mask_binary)
        if len(props) == 0: return np.zeros(84)
        stat = props[0]

        object_mask_float = mask_binary.astype(float)
        object_mask_float[object_mask_float==0] = np.nan
        nuclei_np_object = crop_img * np.dstack([object_mask_float]*3)
        nuclei_np_object_grey = self.rgb2gray(nuclei_np_object).astype(float)
        nuclei_np_object_grey[np.isnan(nuclei_np_object[...,0])] = np.nan

        # 1. Color
        stat_color = {}
        if np.all(np.isnan(nuclei_np_object_grey)):
             color_list = [0]*16
        else:
            color_list = []
            color_list.extend([np.nanmean(nuclei_np_object_grey), np.nanstd(nuclei_np_object_grey), 
                               np.nanmin(nuclei_np_object_grey), np.nanmax(nuclei_np_object_grey)])
            for i in range(3):
                c_data = nuclei_np_object[:,:,i]
                color_list.extend([np.nanmean(c_data), np.nanstd(c_data), np.nanmin(c_data), np.nanmax(c_data)])

        # 2. Cytoplasm
        stat_cyto = self.get_cytoplasm_features(crop_img, mask_binary, all_objects_mask)

        # 3. Morphology
        morph_list = [
            stat['axis_major_length'], stat['axis_minor_length'], 
            stat['axis_major_length'] / (stat['axis_minor_length'] + 1e-8),
            stat['orientation'], stat['orientation'] * (180/np.pi) + 90,
            stat['area'], stat['extent'], stat['solidity'], stat['convex_area'],
            stat['eccentricity'], stat['equivalent_diameter'], stat['perimeter']
        ]
        morph_list.append(stat['perimeter_crofton'] if hasattr(stat, 'perimeter_crofton') else stat['perimeter'])
        
        # 4. Haralick
        stat_haralick = self.get_haralick_features(nuclei_np_object, stat['area'])
        
        # 5,6,7. HistomicsTK (Gradient, Intensity, FSD)
        im_intensity = self.rgb2gray(crop_img)
        
        if HAS_HISTOMICSTK:
            try:
                df_grad = compute_gradient_features.compute_gradient_features(mask_binary, im_intensity, num_hist_bins=10, rprops=[stat])
                list_gradient = list(df_grad.values.reshape(-1))
            except: list_gradient = [0] * 8
            
            try:
                df_int = compute_intensity_features.compute_intensity_features(mask_binary, im_intensity, num_hist_bins=10, rprops=[stat], feature_list=None)
                list_intensity = list(df_int.values.reshape(-1))
            except: list_intensity = [0] * 12
            
            try:
                df_fsd = compute_fsd_features.compute_fsd_features(mask_binary, K=128, Fs=6, Delta=8, rprops=[stat])
                list_fsd = list(df_fsd.values.reshape(-1))
            except:
                list_fsd = [0] * 6
        else:
            list_gradient = [0] * 8
            list_intensity = [0] * 12
            list_fsd = [0] * 6

        x_vec = color_list + list(stat_cyto.values()) + morph_list + list(stat_haralick.values()) + \
                list_gradient + list_intensity + list_fsd
        return np.array(x_vec, dtype=np.float32)

class SpatialFeatureExtractor:
    def __init__(self, base_features, centroids):
        self.raw_feats = base_features
        self.centroids = centroids
        scaler = StandardScaler()
        if len(base_features) > 0:
            self.scaled_feats = scaler.fit_transform(base_features)
        else:
            self.scaled_feats = base_features
        
        self.indices = {
            'Color': slice(0, 16), 'Color_cytoplasm': slice(16, 38), 'Morphology': slice(38, 51),
            'Haralick': slice(51, 58), 'Gradient': slice(58, 66), 'Intensity': slice(66, 78), 'FSD': slice(78, 84)
        }
        self.idx_area = 38 + 5
        self.idx_hete = 51 + 6
        self.idx_orient = 38 + 3
        self.idx_grey = 0
        self.idx_cyto_grey = 16 + 6

    def get_feature_names_spatial(self):
        names = ['dist.mean', 'dist.std', 'dist.min', 'dist.max']
        groups = ['Color', 'Morphology', 'Color_cytoplasm', 'Haralick', 'Gradient', 'Intensity', 'FSD']
        for g in groups:
            for s in ['mean', 'std', 'min', 'max']:
                names.append(f'cosine.{g}.{s}')
        nb_attrs = ['area', 'heterogeneity', 'orientation', 'Grey_mean', 'cyto_Grey_mean', 'Polar.phi']
        for attr in nb_attrs:
            names.append(f'neighbour.{attr}.mean')
            names.append(f'neighbour.{attr}.std')
        return names

    def compute(self):
        N = len(self.centroids)
        if N < 4: return np.zeros((N, 44), dtype=np.float32)
        try: tri = Delaunay(self.centroids)
        except: return np.zeros((N, 44), dtype=np.float32)
            
        indptr, indices = tri.vertex_neighbor_vertices
        spatial_feats = np.zeros((N, 44), dtype=np.float32)
        
        for i in range(N):
            nbr_ids = indices[indptr[i]:indptr[i+1]]
            if len(nbr_ids) == 0: continue
            loc_source = self.centroids[i]
            loc_nbrs = self.centroids[nbr_ids]
            dists = np.linalg.norm(loc_nbrs - loc_source, axis=1)
            valid_mask = dists <= DELAUNAY_DIST_THRESH
            if not np.any(valid_mask): continue
            dists = dists[valid_mask]
            nbr_ids = nbr_ids[valid_mask]
            loc_nbrs = loc_nbrs[valid_mask]
            
            spatial_feats[i, 0:4] = [np.mean(dists), np.std(dists), np.min(dists), np.max(dists)]
            current_col = 4
            for group_name, sl in self.indices.items():
                vec_source = self.scaled_feats[i, sl].reshape(1, -1)
                vec_nbrs = self.scaled_feats[nbr_ids, sl]
                dot_prod = np.sum(vec_source * vec_nbrs, axis=1)
                norm_source = np.linalg.norm(vec_source)
                norm_nbrs = np.linalg.norm(vec_nbrs, axis=1)
                denom = norm_source * norm_nbrs
                cos_sims = np.divide(dot_prod, denom, out=np.zeros_like(dot_prod), where=denom!=0)
                spatial_feats[i, current_col:current_col+4] = [np.mean(cos_sims), np.std(cos_sims), np.min(cos_sims), np.max(cos_sims)]
                current_col += 4
            rel_loc = loc_nbrs - loc_source
            phi = np.arctan2(rel_loc[:,1], rel_loc[:,0])
            nb_indices = [self.idx_area, self.idx_hete, self.idx_orient, self.idx_grey, self.idx_cyto_grey]
            for idx in nb_indices:
                vals = self.scaled_feats[nbr_ids, idx]
                spatial_feats[i, current_col:current_col+2] = [np.mean(vals), np.std(vals)]
                current_col += 2
            spatial_feats[i, current_col:current_col+2] = [np.mean(phi), np.std(phi)]
        return np.nan_to_num(spatial_feats)

# ==============================
# Utilities & Parsers
# ==============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_cells_json(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def parse_cells_and_contours(data: Dict, scale: float) -> Dict[str, np.ndarray]:
    """
    Parses cells to get bboxes, centroids, and tracks contours for mask generation.
    Returns dictionary with:
      - 'bbox': [S, 4]
      - 'centroid': [S, 2]
      - 'contour': List of [N, 2] arrays
    """
    cells = data.get("cells", [])
    S = len(cells)

    bboxes = np.zeros((S, 4), dtype=np.int32)
    centroids = np.zeros((S, 2), dtype=np.float32)
    contours = []

    for i, c in enumerate(cells):
        # Handle bbox: try generic JSON format [[ymin, xmin], [ymax, xmax]]
        # If your JSON has pre-doubled [2*ymin...], handle accordingly.
        # Here we assume standard format relative to original image, multiplied by scale.
        (ymin, xmin), (ymax, xmax) = c["bbox"]
        
        # NOTE: If your JSON "bbox" entries are already doubled (like in Script 1's comments),
        # but you set RESIZE_SCALE=1, this math holds.
        # If they are original coords, they get scaled here.
        bboxes[i, 0] = int(ymin * scale)
        bboxes[i, 1] = int(ymax * scale)
        bboxes[i, 2] = int(xmin * scale)
        bboxes[i, 3] = int(xmax * scale)

        cx, cy = c["centroid"]
        centroids[i, 0] = float(cx * scale)
        centroids[i, 1] = float(cy * scale)
        
        # Get contour for instance map
        if 'contour' in c:
            cnt = np.array(c['contour']) * scale
            contours.append(cnt.astype(np.int32))
        else:
            # Fallback if no contour: make a box contour
            box_cnt = np.array([
                [bboxes[i, 2], bboxes[i, 0]], # xmin, ymin
                [bboxes[i, 3], bboxes[i, 0]], # xmax, ymin
                [bboxes[i, 3], bboxes[i, 1]], # xmax, ymax
                [bboxes[i, 2], bboxes[i, 1]]  # xmin, ymax
            ], dtype=np.int32)
            contours.append(box_cnt)

    return {"bbox": bboxes, "centroid": centroids, "contour": contours}

def generate_instance_map(shape: Tuple[int, int], contours: List[np.ndarray]) -> np.ndarray:
    """
    Creates an instance map from contours.
    shape: (H, W)
    """
    inst_map = np.zeros(shape, dtype=np.int32)
    for i, contour in enumerate(contours):
        cell_id = i + 1
        cv2.fillPoly(inst_map, [contour], color=cell_id)
    return inst_map

# ==============================
# Geometry transforms
# ==============================
def resize_image(img: Image.Image, scale: float = RESIZE_SCALE) -> Image.Image:
    if scale == 1.0: return img
    W, H = img.size
    new_W, new_H = int(round(W * scale)), int(round(H * scale))
    return img.resize((new_W, new_H), resample=Image.BICUBIC)

def pad_data_and_mask(img: Image.Image, inst_map: np.ndarray, multiple: int = PATCH_SIZE) -> Tuple[Image.Image, np.ndarray]:
    """
    Pads image (white) and instance map (zero) to multiples of PATCH_SIZE.
    """
    W, H = img.size
    target_W = int(math.ceil(W / multiple) * multiple)
    target_H = int(math.ceil(H / multiple) * multiple)

    pad_right = target_W - W
    pad_bottom = target_H - H

    if pad_right == 0 and pad_bottom == 0:
        return img, inst_map

    # Pad Image
    canvas = Image.new("RGB", (target_W, target_H), color=(255, 255, 255))
    canvas.paste(img, (0, 0))
    
    # Pad Instance Map
    inst_padded = np.pad(inst_map, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
    
    return canvas, inst_padded

# ==============================
# Global graph construction
# ==============================
def build_global_graph(bboxes: np.ndarray, centroids: np.ndarray, features: np.ndarray, k: int = K_NEIGHBORS) -> Data:
    """
    Build KNN graph. Now includes 'x' (features) and 'node_id'.
    """
    pos = torch.from_numpy(centroids).float()    # [S, 2]
    x_bbox = torch.from_numpy(bboxes).int()      # [S, 4]
    x_feat = torch.from_numpy(features).float()  # [S, 128]
    
    edge_index = knn_graph(pos, k=k, loop=False)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        x_bbox=x_bbox,
        x_centroid=pos,
        # Create a sequential node ID matching the JSON list order
        node_id=torch.arange(len(centroids)).long()
    )
    return data

# ==============================
# Patch tiling + subgraph extraction
# ==============================
def iter_patch_windows(W: int, H: int, patch: int = PATCH_SIZE) -> List[Tuple[int, int, int, int]]:
    windows = []
    for y0 in range(0, H, patch):
        for x0 in range(0, W, patch):
            windows.append((x0, y0, x0 + patch, y0 + patch))
    return windows

def nodes_in_window(centroids: torch.Tensor,
                    x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    xs = centroids[:, 0]
    ys = centroids[:, 1]
    mask = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return torch.nonzero(mask, as_tuple=False).view(-1)

def extract_patch_subgraph(global_data: Data,
                           window: Tuple[int, int, int, int],
                           center: Tuple[int, int]) -> Data:
    x0, y0, x1, y1 = window
    cx, cy = center

    node_idx = nodes_in_window(global_data.x_centroid, x0, y0, x1, y1)
    if node_idx.numel() == 0:
        sub = Data(num_nodes=0, edge_index=torch.empty(2, 0, dtype=torch.long))
        sub.patch_center = torch.tensor([[cx, cy]], dtype=torch.int32)
        sub.window = torch.tensor([x0, y0, x1, y1], dtype=torch.int32)
        return sub

    eidx, _ = subgraph(node_idx, global_data.edge_index, relabel_nodes=True)

    # Note: global_data.x is now included
    sub = Data(
        num_nodes=node_idx.shape[0],
        x=global_data.x[node_idx],
        edge_index=eidx,
        x_bbox=global_data.x_bbox[node_idx],
        x_centroid=global_data.x_centroid[node_idx],
        node_id=global_data.node_id[node_idx]
    )
    sub.patch_center = torch.tensor([[cx, cy]], dtype=torch.int32)
    sub.window = torch.tensor([x0, y0, x1, y1], dtype=torch.int32)
    return sub

def save_subgraph(sub: Data, base_name: str, center: Tuple[int, int], out_dir: str):
    if sub.num_nodes is None or sub.num_nodes == 0:
        return
    cx, cy = center
    out_name = f"{base_name}_{cx}x{cy}.pt"
    ensure_dir(out_dir)
    torch.save(sub, os.path.join(out_dir, out_name))

# ==============================
# Pairing and Processing
# ==============================
def find_pairs(img_dir: str, json_dir: str) -> List[Tuple[str, str]]:
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    json_by_key = {}
    for jp in glob.glob(os.path.join(json_dir, "*.json")):
        s = stem(jp)
        key = s.replace("_cells", "")
        json_by_key[key] = jp

    pairs = []
    for ip in img_paths:
        s = stem(ip)
        if s in json_by_key:
            pairs.append((ip, json_by_key[s]))
    return pairs

def process_one(image_path: str, json_path: str):
    base = stem(image_path)

    # 1) Load image, resize
    img = load_image(image_path)
    img_resized = resize_image(img, scale=RESIZE_SCALE)
    W_rz, H_rz = img_resized.size

    # 2) Load JSON, parse Metadata AND Contours
    data = load_cells_json(json_path)
    # This parses bboxes and centroids, and scales contours
    ann = parse_cells_and_contours(data, scale=RESIZE_SCALE)

    # 3) Generate Instance Map (Needed for visual feature extraction)
    # Using H_rz, W_rz from the resized image
    inst_map = generate_instance_map((H_rz, W_rz), ann['contour'])

    # 4) Pad image and instance map
    img_padded, inst_map_padded = pad_data_and_mask(img_resized, inst_map, multiple=PATCH_SIZE)
    img_padded_np = np.array(img_padded)
    Wp, Hp = img_padded.size

    # 5) Feature Extraction - Part A: Base (Visual) Features
    extractor = NucleiFeatureExtractor()
    base_feats_list = []
    
    # Iterate through cells
    # Note: IDs in inst_map start at 1, matching the index + 1
    bboxes = ann["bbox"]
    for i in tqdm(range(len(bboxes))):
        nid = i + 1
        # Extract 84 visual features
        f = extractor.process_single_nucleus(img_padded_np, inst_map_padded, nid, bboxes[i])
        base_feats_list.append(f)
    
    base_feats = np.array(base_feats_list) if len(base_feats_list) > 0 else np.zeros((0, 84))

    # 6) Feature Extraction - Part B: Spatial Features
    spatial_extractor = SpatialFeatureExtractor(base_feats, ann["centroid"])
    spatial_feats = spatial_extractor.compute() # [N, 44]

    # 7) Concatenate to [N, 128]
    total_feats = np.hstack([base_feats, spatial_feats])

    # 8) Build Global Graph (with features)
    global_graph = build_global_graph(ann["bbox"], ann["centroid"], total_feats, k=K_NEIGHBORS)

    # 9) Tile Patches & Save
    windows = iter_patch_windows(Wp, Hp, patch=PATCH_SIZE)
    for x0, y0, x1, y1 in windows:
        center = (x0 + PATCH_SIZE // 2, y0 + PATCH_SIZE // 2)
        sub = extract_patch_subgraph(global_graph, (x0, y0, x1, y1), center)
        save_subgraph(sub, base, center, GRAPH_OUT_DIR)

# ==============================
# Main
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-patch graphs from images and CellViT segmentation JSON."
    )
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Dataset directory. Also used for feature_list_128_inferred.csv")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="PNG directory (default: <base_dir>/images)")
    parser.add_argument("--json_dir", type=str, default=None,
                        help="Cell segmentation JSON directory (default: <base_dir>/cell_segmentation)")
    parser.add_argument("--graph_out", type=str, default=None,
                        help="Graph output (default: <base_dir>/graphs_cellvit_sam_inferred4UNI_histomic)")
    return parser.parse_args()


def main():
    global BASE_DIR, IMG_DIR, JSON_DIR, GRAPH_OUT_DIR
    args = parse_args()
    BASE_DIR = args.base_dir
    IMG_DIR = args.img_dir or os.path.join(args.base_dir, "images")
    JSON_DIR = args.json_dir or os.path.join(args.base_dir, "cell_segmentation")
    GRAPH_OUT_DIR = args.graph_out or os.path.join(args.base_dir, "graphs_cellvit_sam_inferred4UNI_histomic")

    ensure_dir(GRAPH_OUT_DIR)
    
    # 0. Generate Feature List CSV (Requirement)
    extractor = NucleiFeatureExtractor()
    dummy_feats = np.zeros((2, 84))
    dummy_cents = np.zeros((2, 2))
    spatial_extractor = SpatialFeatureExtractor(dummy_feats, dummy_cents)
    all_names = extractor.get_feature_names_base() + spatial_extractor.get_feature_names_spatial()
    csv_path = os.path.join(BASE_DIR, 'feature_list_128_inferred.csv')
    pd.DataFrame(all_names, columns=['Feature_Name']).to_csv(csv_path)
    print(f"Feature list saved to {csv_path} (Count: {len(all_names)})")

    # 1. Process Pairs
    pairs = find_pairs(IMG_DIR, JSON_DIR)
    if not pairs:
        print("No image/json pairs found; check directories and naming.")
        return

    for img_path, json_path in pairs:
        print(f"Processing: {os.path.basename(img_path)}")
        process_one(img_path, json_path)

    print("Done.")


if __name__ == "__main__":
    main()