"""
Cell type inference on a whole-slide image.

Memory strategy:
  - Never load the full WSI into RAM.
  - Keep only cell bboxes/centroids from JSON (no slide-wide KNN graph).
  - Build a local KNN subgraph per patch for cells inside that window.
  - Read only the native pixel region needed for each 224x224 patch (tifffile / czifile).
  - Stream patch -> infer -> discard; skip patches with zero cells.
  - DataLoader workers prefetch patches in parallel (predict.num_workers).
"""

import argparse
import gc
import json
import math
import os
import shutil
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
from tqdm import tqdm

from model import ColorCell
from data import get_trans_func

import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

Image.MAX_IMAGE_PIXELS = None

PATCH_SIZE = 224
K_NEIGHBORS = 12
TARGET_MPP = 0.25
# PNG above this pixel count requires pyvips for lazy reads (PIL decodes the full image).
PNG_LAZY_PIXEL_LIMIT = 4096 * 4096


# ---------------------------------------------------------------------------
# Array / image helpers
# ---------------------------------------------------------------------------
def extract_rgb_image(image_array: np.ndarray) -> np.ndarray:
    """Extract an (H, W, 3) RGB array from a czifile-style array."""
    arr = np.asarray(image_array)

    if arr.ndim == 5 and arr.shape[-1] == 3:
        return arr[0, 0]
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)

    raise ValueError(
        f"Unsupported array shape {arr.shape}; expected RGB with trailing dim 3"
    )


def _array_to_uint8_rgb(image_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_array)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)

    return arr


def compute_padded_layout(native_w: int, native_h: int, resize_scale: float) -> Tuple[int, int, int, int]:
    """Return (resized_w, resized_h, padded_w, padded_h) in model coordinate space."""
    resized_w = int(round(native_w * resize_scale))
    resized_h = int(round(native_h * resize_scale))
    padded_w = int(math.ceil(resized_w / PATCH_SIZE) * PATCH_SIZE)
    padded_h = int(math.ceil(resized_h / PATCH_SIZE) * PATCH_SIZE)
    return resized_w, resized_h, padded_w, padded_h


def compose_patch_from_native_region(
    native_region: np.ndarray,
    out_w: int,
    out_h: int,
    paste_x: int,
    paste_y: int,
    patch_size: int = PATCH_SIZE,
) -> Image.Image:
    """Resize a native ROI to (out_w, out_h) and paste into a white 224x224 patch."""
    out = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
    if out_w <= 0 or out_h <= 0:
        return out

    rgb = _array_to_uint8_rgb(native_region)
    part = Image.fromarray(rgb).resize((out_w, out_h), resample=Image.BICUBIC)
    out.paste(part, (paste_x, paste_y))
    return out


# ---------------------------------------------------------------------------
# Lazy WSI patch readers (no full-slide load)
# ---------------------------------------------------------------------------
class WSIPatchSource(ABC):
    """Read 224x224 patches in padded/resized coordinate space without loading the full WSI."""

    def __init__(self, path: str, resize_scale: float):
        self.path = path
        self.resize_scale = resize_scale
        self.native_w, self.native_h = self._read_native_size()
        self.resized_w, self.resized_h, self.padded_w, self.padded_h = compute_padded_layout(
            self.native_w, self.native_h, resize_scale
        )

    @abstractmethod
    def _read_native_size(self) -> Tuple[int, int]:
        """Return (width, height) of the WSI at the level referenced by slide_mpp."""

    @abstractmethod
    def _read_native_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read native pixels in [x, x+w) x [y, y+h). Origin is top-left."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_patch(self, x0: int, y0: int, patch_size: int = PATCH_SIZE) -> Image.Image:
        """
        Read one patch at top-left (x0, y0) in padded/resized coordinate space.
        Matches crop(resize(pad(WSI))) semantics used during training.
        """
        ix0 = max(x0, 0)
        iy0 = max(y0, 0)
        ix1 = min(x0 + patch_size, self.resized_w)
        iy1 = min(y0 + patch_size, self.resized_h)

        if ix0 >= ix1 or iy0 >= iy1:
            return Image.new("RGB", (patch_size, patch_size), (255, 255, 255))

        out_w, out_h = ix1 - ix0, iy1 - iy0
        paste_x, paste_y = ix0 - x0, iy0 - y0
        scale = self.resize_scale

        nx0 = ix0 / scale
        ny0 = iy0 / scale
        nx1 = ix1 / scale
        ny1 = iy1 / scale

        x_start = int(math.floor(nx0))
        y_start = int(math.floor(ny0))
        x_end = int(math.ceil(nx1))
        y_end = int(math.ceil(ny1))
        x_end = max(x_end, x_start + 1)
        y_end = max(y_end, y_start + 1)

        native = self._read_native_region(x_start, y_start, x_end - x_start, y_end - y_start)
        return compose_patch_from_native_region(
            native, out_w, out_h, paste_x, paste_y, patch_size=patch_size
        )


class TiffPatchSource(WSIPatchSource):
    """Region reads for .tif / .tiff / .svs via tifffile (tiled pages decode only needed tiles)."""

    def __init__(self, path: str, resize_scale: float):
        self._tif: Optional[tifffile.TiffFile] = None
        self._page = None
        self._zarr = None
        super().__init__(path, resize_scale)

    def _ensure_open(self) -> None:
        if self._tif is None:
            import zarr

            self._tif = tifffile.TiffFile(self.path)
            self._page = self._tif.pages[0]
            # Lazy region reads: indexing the zarr store only decodes the tiles
            # that overlap the requested slice (tiled TIFF). TiffPage.asarray has
            # no spatial-crop arg ('key' selects pages/frames, not a region).
            self._zarr = zarr.open(self._page.aszarr(), mode="r")

    def _read_native_size(self) -> Tuple[int, int]:
        self._ensure_open()
        page = self._page
        if hasattr(page, "shape") and len(page.shape) >= 2:
            h, w = int(page.shape[0]), int(page.shape[1])
        else:
            h, w = int(page.imagelength), int(page.imagewidth)
        return w, h

    def _read_native_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        self._ensure_open()
        img_w, img_h = self.native_w, self.native_h

        x0 = max(0, min(x, img_w))
        y0 = max(0, min(y, img_h))
        x1 = max(x0, min(x + w, img_w))
        y1 = max(y0, min(y + h, img_h))

        if x0 >= x1 or y0 >= y1:
            return np.full((max(1, h), max(1, w), 3), 255, dtype=np.uint8)

        region = np.asarray(self._zarr[y0:y1, x0:x1])
        # Drop any leading singleton axes (e.g. shape (1, h, w, 3)) so the result
        # is (h, w) or (h, w, C) as expected by the RGB extraction helpers.
        while region.ndim > 3 and region.shape[0] == 1:
            region = region[0]
        return region

    def close(self) -> None:
        if self._tif is not None:
            self._tif.close()
            self._tif = None
            self._page = None
            self._zarr = None


class CziPatchSource(WSIPatchSource):
    """Region reads for .czi by decoding only subblocks overlapping the ROI (czifile 2019+)."""

    def __init__(self, path: str, resize_scale: float):
        self._czi = None
        self._plane_starts: Dict[str, int] = {}
        super().__init__(path, resize_scale)

    def _ensure_open(self) -> None:
        if self._czi is None:
            import czifile

            self._czi = czifile.CziFile(self.path)
            for axis, start in zip(self._czi.axes, self._czi.start):
                if axis != "M":
                    self._plane_starts[axis] = int(start)

    def _read_native_size(self) -> Tuple[int, int]:
        self._ensure_open()
        czi = self._czi
        shape = czi.shape
        axes = [a for a in czi.axes if a != "M"]
        size_map = {ax: int(shape[i]) for i, ax in enumerate(axes)}
        return size_map["X"], size_map["Y"]

    @staticmethod
    def _entry_dim_map(entry) -> Dict[str, object]:
        return {dim.dimension: dim for dim in entry.dimension_entries}

    def _entry_matches_default_plane(self, entry) -> bool:
        dims = self._entry_dim_map(entry)
        for axis, start in self._plane_starts.items():
            if axis in ("X", "Y"):
                continue
            if axis not in dims:
                continue
            if int(dims[axis].start) != int(start):
                return False
        return True

    def _read_native_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        self._ensure_open()
        czi = self._czi
        roi_x1, roi_y1 = x + w, y + h
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        x_off = self._plane_starts.get("X", 0)
        y_off = self._plane_starts.get("Y", 0)

        for entry in czi.filtered_subblock_directory:
            if not self._entry_matches_default_plane(entry):
                continue

            dims = self._entry_dim_map(entry)
            if "X" not in dims or "Y" not in dims:
                continue

            # Subblock positions are absolute; mosaic array is 0-based after subtracting czi.start.
            ex = int(dims["X"].start) - x_off
            ey = int(dims["Y"].start) - y_off
            ew = int(dims["X"].size)
            eh = int(dims["Y"].size)
            ex1, ey1 = ex + ew, ey + eh

            ox0 = max(x, ex)
            oy0 = max(y, ey)
            ox1 = min(roi_x1, ex1)
            oy1 = min(roi_y1, ey1)
            if ox0 >= ox1 or oy0 >= oy1:
                continue

            tile = np.asarray(entry.data_segment().data(resize=True))
            tile_rgb = extract_rgb_image(tile)

            sx0, sy0 = ox0 - ex, oy0 - ey
            tw, th = ox1 - ox0, oy1 - oy0
            out[oy0 - y : oy0 - y + th, ox0 - x : ox0 - x + tw] = tile_rgb[sy0 : sy0 + th, sx0 : sx0 + tw]

        return out

    def close(self) -> None:
        if self._czi is not None:
            self._czi.close()
            self._czi = None


class PngPatchSource(WSIPatchSource):
    """
    PNG region reads.

    Uses pyvips when installed (true lazy decode). Otherwise falls back to PIL for
    small images only; large PNGs raise a clear error instead of OOM.
    """

    def __init__(self, path: str, resize_scale: float):
        self._pil_image: Optional[Image.Image] = None
        self._pyvips_image = None
        self._use_pyvips = False
        super().__init__(path, resize_scale)

    def _read_native_size(self) -> Tuple[int, int]:
        try:
            import pyvips

            self._pyvips_image = pyvips.Image.new_from_file(self.path, access="random")
            self._use_pyvips = True
            return int(self._pyvips_image.width), int(self._pyvips_image.height)
        except ImportError:
            pass

        with Image.open(self.path) as img:
            w, h = img.size
        if w * h > PNG_LAZY_PIXEL_LIMIT:
            raise MemoryError(
                f"PNG is {w}x{h} ({w * h:,} px). Install pyvips for lazy reads "
                f"(pip install pyvips), or convert to tiled TIFF/SVS first."
            )
        return w, h

    def _read_native_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        img_w, img_h = self.native_w, self.native_h
        x0 = max(0, min(x, img_w))
        y0 = max(0, min(y, img_h))
        x1 = max(x0, min(x + w, img_w))
        y1 = max(y0, min(y + h, img_h))

        if x0 >= x1 or y0 >= y1:
            return np.full((max(1, h), max(1, w), 3), 255, dtype=np.uint8)

        if self._use_pyvips:
            region = self._pyvips_image.crop(x0, y0, x1 - x0, y1 - y0)
            if region.bands == 4:
                region = region.flatten(background=[255, 255, 255])
            elif region.bands == 1:
                region = region.bandjoin([region, region, region])
            mem = region.write_to_memory()
            arr = np.frombuffer(mem, dtype=np.uint8).reshape(y1 - y0, x1 - x0, region.bands)
            return arr[..., :3]

        if self._pil_image is None:
            self._pil_image = Image.open(self.path).convert("RGB")
        crop = self._pil_image.crop((x0, y0, x1, y1))
        return np.asarray(crop)

    def close(self) -> None:
        if self._pil_image is not None:
            self._pil_image.close()
            self._pil_image = None
        self._pyvips_image = None


def open_wsi_patch_source(wsi_path: str, slide_mpp: float) -> WSIPatchSource:
    ext = os.path.splitext(wsi_path)[1].lower()
    resize_scale = slide_mpp / TARGET_MPP

    if ext in (".tif", ".tiff", ".svs"):
        return TiffPatchSource(wsi_path, resize_scale)
    if ext == ".czi":
        return CziPatchSource(wsi_path, resize_scale)
    if ext == ".png":
        return PngPatchSource(wsi_path, resize_scale)

    raise ValueError(
        f"Unsupported WSI format '{ext}'; expected .png, .tif, .tiff, .svs, or .czi"
    )


# ---------------------------------------------------------------------------
# Cell JSON + graph
# ---------------------------------------------------------------------------
def load_cells_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_cells(data: Dict) -> Dict[str, np.ndarray]:
    cells = data.get("cells", [])
    s = len(cells)

    bboxes = np.zeros((s, 4), dtype=np.int32)
    centroids = np.zeros((s, 2), dtype=np.float32)

    for i, c in enumerate(cells):
        (ymin2, xmin2), (ymax2, xmax2) = c["bbox"]
        bboxes[i, 0] = int(ymin2)
        bboxes[i, 1] = int(ymax2)
        bboxes[i, 2] = int(xmin2)
        bboxes[i, 3] = int(xmax2)

        cx2, cy2 = c["centroid"]
        centroids[i, 0] = float(cx2)
        centroids[i, 1] = float(cy2)

    return {"bbox": bboxes, "centroid": centroids}


def load_cell_annotations(cell_seg_path: str) -> Dict[str, np.ndarray]:
    """Load per-cell bboxes and centroids (no graph)."""
    return parse_cells(load_cells_json(cell_seg_path))


def build_patch_cell_index(
    ann: Dict[str, np.ndarray], padded_w: int, padded_h: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """Map each patch top-left (x0, y0) to the cell indices inside that patch."""
    centroids = ann["centroid"]
    patch_to_ids: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    max_x_idx = max((padded_w // PATCH_SIZE) - 1, 0)
    max_y_idx = max((padded_h // PATCH_SIZE) - 1, 0)

    for idx, (cx, cy) in enumerate(centroids):
        x_idx = int(cx) // PATCH_SIZE
        y_idx = int(cy) // PATCH_SIZE
        if x_idx < 0 or y_idx < 0 or x_idx > max_x_idx or y_idx > max_y_idx:
            continue
        patch_to_ids[(x_idx * PATCH_SIZE, y_idx * PATCH_SIZE)].append(idx)

    return {
        patch_xy: np.asarray(cell_ids, dtype=np.int64)
        for patch_xy, cell_ids in patch_to_ids.items()
        if len(cell_ids) > 0
    }


def build_patch_subgraph(
    ann: Dict[str, np.ndarray],
    node_idx: np.ndarray,
    center: Tuple[int, int],
    k: int = K_NEIGHBORS,
) -> Optional[Data]:
    """
    Build a KNN subgraph for cells in one patch.
    """
    cx, cy = center

    if node_idx.size == 0:
        return None

    patch_centroid = torch.from_numpy(ann["centroid"][node_idx]).float()
    patch_bbox = torch.from_numpy(ann["bbox"][node_idx]).int()
    n_nodes = patch_centroid.shape[0]

    k_eff = min(k, max(n_nodes - 1, 0))
    if k_eff > 0:
        edge_index = knn_graph(patch_centroid, k=k_eff, loop=False)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    sub = Data(
        num_nodes=n_nodes,
        edge_index=edge_index,
        x_bbox=patch_bbox,
        x_centroid=patch_centroid,
    )
    sub.patch_center = torch.tensor([[cx, cy]], dtype=torch.int32)
    return sub


def _loader_worker_init_fn(_worker_id: int) -> None:
    """Limit per-worker CPU threads to reduce oversubscription with many workers."""
    torch.set_num_threads(1)


class WSIPatchCellDataset(Dataset):
    """Dataset over cell-containing patch coordinates; each worker opens its own WSI handle."""

    def __init__(
        self,
        wsi_path: str,
        slide_mpp: float,
        ann: Dict[str, np.ndarray],
        patch_keys: List[Tuple[int, int]],
        patch_cell_index: Dict[Tuple[int, int], np.ndarray],
        slide_id: str,
        cell_enc_type: str,
    ):
        self.wsi_path = wsi_path
        self.slide_mpp = slide_mpp
        self.ann = ann
        self.patch_keys = patch_keys
        self.patch_cell_index = patch_cell_index
        self.slide_id = slide_id
        self.tile_transform = get_trans_func(cell_enc_type=cell_enc_type)
        self._patch_source: Optional[WSIPatchSource] = None

    def _get_patch_source(self) -> WSIPatchSource:
        if self._patch_source is None:
            self._patch_source = open_wsi_patch_source(self.wsi_path, self.slide_mpp)
        return self._patch_source

    def __len__(self) -> int:
        return len(self.patch_keys)

    def __getitem__(self, idx: int):
        x0, y0 = self.patch_keys[idx]
        center = (x0 + PATCH_SIZE // 2, y0 + PATCH_SIZE // 2)
        sub = build_patch_subgraph(self.ann, self.patch_cell_index[(x0, y0)], center)
        tile = self._get_patch_source().read_patch(x0, y0)
        tile = self.tile_transform(tile)
        return sub, tile, self.slide_id


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------
def collect_predictions_from_batch(
    model: ColorCell,
    data_batch: Batch,
    tile_batch: torch.Tensor,
    slide_id,
    device: torch.device,
    graph_type: str,
    mode: str,
    results_acc: List[dict],
) -> None:
    data_batch = data_batch.to(device, non_blocking=device.type == "cuda")
    tile_batch = tile_batch.to(device, non_blocking=device.type == "cuda")

    with torch.no_grad():
        if mode in ("MLP", "ensemble"):
            pred = model(tile_batch, data_batch)
        elif mode in ("TabNet", "both"):
            pred, _ = model(tile_batch, data_batch)
        else:
            raise ValueError(f"Unsupported model mode: {mode}")

        prob = F.softmax(pred, dim=-1)

        if graph_type == "patch_subgraph":
            for i in range(prob.shape[0]):
                results_acc.append(
                    {
                        "centroid": data_batch.x_centroid[i].tolist(),
                        "type_prob": prob[i][torch.argmax(prob[i])].item(),
                        "type": torch.argmax(prob[i]).item(),
                        "prob_vector": prob[i].tolist(),
                    }
                )
        elif graph_type == "k_hop_subgraph":
            global_mapping = data_batch.ptr[:-1] + data_batch.mapping
            for i in global_mapping.tolist():
                results_acc.append(
                    {
                        "centroid": data_batch.x_centroid[i].tolist(),
                        "type_prob": prob[i][torch.argmax(prob[i])].item(),
                        "type": torch.argmax(prob[i]).item(),
                        "prob_vector": prob[i].tolist(),
                    }
                )
        else:
            raise ValueError(f"Unsupported graph_type: {graph_type}")


def load_colorcell_model(cfg: dict) -> ColorCell:
    ckpt_dir = cfg["colorcell_ckpt"]["save_dir"]
    n_steps = cfg["model"].get("n_steps", 1)
    gamma = cfg["model"].get("gamma", 1.5)
    pre_mlp_hidden_dim = cfg["model"].get("pre_mlp_hidden_dim", 256)
    pre_tabnet_hidden_dim = cfg["model"].get("pre_tabnet_hidden_dim", 32)

    model = ColorCell(
        cell_enc_type=cfg["model"]["cell_enc_type"],
        cell_emb_dim=cfg["model"]["cell_emb_dim"],
        mlp_hidden_dims=cfg["model"]["mlp_hidden_dims"],
        output_dim=cfg["model"]["output_dim"],
        norm_type=cfg["model"]["norm_type"],
        act=nn.LeakyReLU(),
        cell_enc_dir=cfg["model"]["cell_enc_dir"],
        mode=cfg["model"]["mode"],
        n_steps=n_steps,
        gamma=gamma,
        pre_mlp_hidden_dim=pre_mlp_hidden_dim,
        pre_tabnet_hidden_dim=pre_tabnet_hidden_dim,
    )

    state_dict = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    if cfg["model_mode"] == "LP":
        model.classifier.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    return model


def streaming_inference(
    wsi_path: str,
    slide_mpp: float,
    padded_w: int,
    padded_h: int,
    ann: Dict[str, np.ndarray],
    patch_cell_index: Dict[Tuple[int, int], np.ndarray],
    slide_id: str,
    cfg: dict,
    outdir: str,
    config_path: str,
) -> None:
    """DataLoader over cell patches: parallel CPU prefetch + batched GPU inference."""
    predict_cfg = cfg.get("predict", cfg.get("testing", {}))
    batch_size = predict_cfg["batch_size"]
    num_workers = int(predict_cfg.get("num_workers", 0))
    graph_type = cfg["model"]["graph_type"]
    mode = cfg["model"]["mode"]

    patch_keys = [
        (x0, y0)
        for x0, y0 in sorted(patch_cell_index.keys(), key=lambda p: (p[1], p[0]))
        if x0 + PATCH_SIZE <= padded_w and y0 + PATCH_SIZE <= padded_h
    ]
    n_patches_with_cells = len(patch_keys)

    if n_patches_with_cells == 0:
        raise RuntimeError(
            "No patches with cells were generated. Check slide_mpp, segmentation JSON, and WSI alignment."
        )

    dataset = WSIPatchCellDataset(
        wsi_path=wsi_path,
        slide_mpp=slide_mpp,
        ann=ann,
        patch_keys=patch_keys,
        patch_cell_index=patch_cell_index,
        slide_id=slide_id,
        cell_enc_type=cfg["model"]["cell_enc_type"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "exclude_keys": ["x_expr"],
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = _loader_worker_init_fn

    test_loader = DataLoader(dataset, **loader_kwargs)

    model = load_colorcell_model(cfg)
    model.eval()
    model.to(device)

    results_acc: List[dict] = []
    print(f"  DataLoader: batch_size={batch_size}, num_workers={num_workers}", flush=True)

    for batch in tqdm(test_loader, desc="inference", unit="batch"):
        data, tile, slide_id_batch = batch
        collect_predictions_from_batch(
            model=model,
            data_batch=data,
            tile_batch=tile,
            slide_id=slide_id_batch,
            device=device,
            graph_type=graph_type,
            mode=mode,
            results_acc=results_acc,
        )
        gc.collect()

    print(f"  {n_patches_with_cells} patches with cells, {len(results_acc)} cell predictions")

    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(outdir, "inference_config.yaml"))

    out_path = os.path.join(outdir, f"{slide_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(i + 1): val for i, val in enumerate(results_acc)}, f, indent=4)
    print(f"Wrote {out_path} ({len(results_acc)} cells)")


def main():
    parser = argparse.ArgumentParser(description="Cell type inference on a whole slide image.")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to WSI (.czi, .png, .svs, .tiff)")
    parser.add_argument("--cell_seg_path", type=str, required=True, help="Path to cell segmentation JSON")
    parser.add_argument("--slide_mpp", type=float, required=True, help="Micrometers per pixel of the WSI")
    parser.add_argument("--configs", type=str, required=True, help="Path to YAML config (model + predict)")
    parser.add_argument("--outdir", type=str, required=True, help="Directory for output JSON files")
    args = parser.parse_args()

    with open(args.configs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    slide_id = os.path.splitext(os.path.basename(args.wsi_path))[0]
    resize_scale = args.slide_mpp / TARGET_MPP

    print(f"WSI: {args.wsi_path}", flush=True)
    print(f"Cell seg: {args.cell_seg_path}", flush=True)
    print(f"slide_id: {slide_id}")
    print(f"RESIZE_SCALE (slide_mpp / {TARGET_MPP}): {resize_scale:.6f}", flush=True)
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"), flush=True)

    print("Loading cell annotations from segmentation JSON...", flush=True)
    ann = load_cell_annotations(args.cell_seg_path)
    print(f"  {len(ann['centroid'])} cells", flush=True)

    print("Opening WSI for lazy patch reads (no full-slide load)...")
    with open_wsi_patch_source(args.wsi_path, args.slide_mpp) as patch_source:
        print(f"  native size: {patch_source.native_w} x {patch_source.native_h}", flush=True)
        print(f"  padded grid: {patch_source.padded_w} x {patch_source.padded_h}", flush=True)
        patch_cell_index = build_patch_cell_index(
            ann=ann,
            padded_w=patch_source.padded_w,
            padded_h=patch_source.padded_h,
        )
        print(f"  cell-containing patches: {len(patch_cell_index)}", flush=True)

        print(
            "Streaming inference (DataLoader prefetch -> batched GPU infer)...",
            flush=True,
        )
        streaming_inference(
            wsi_path=args.wsi_path,
            slide_mpp=args.slide_mpp,
            padded_w=patch_source.padded_w,
            padded_h=patch_source.padded_h,
            ann=ann,
            patch_cell_index=patch_cell_index,
            slide_id=slide_id,
            cfg=cfg,
            outdir=args.outdir,
            config_path=args.configs,
        )


if __name__ == "__main__":
    main()
