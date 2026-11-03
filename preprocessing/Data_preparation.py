import argparse
import os
import numpy as np
import cv2
import scipy.io as sio
import scipy.sparse as sp
from PIL import Image
from Load_dataset import load_10x_Xenium_Anndata
from Coord_alignment import coords_alignment, load_transformation_matrix, project_points
from Load_dataset import load_image
import pandas as pd

# Allow PIL to process large whole slide images without throwing a DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

# -------------------------------------------------------------------------
# Define your cell type mapping dictionary here (0-indexed)
# -------------------------------------------------------------------------
cell_type_dict = {
    "Epithelial/Cancer cells": 0,
    "CD4+ T cells": 1,
    "CD8+ T cells": 1,
    "NK cells": 1,
    "B cells": 1,
    "Plasma cells": 2,
    "Mast cells": 3,
    "Monocytes/Macrophages": 4,
    "Dendritic cells": 5,
    "Fibroblasts": 6,
    "Smooth muscle cells/Pericytes": 7,
    "Endothelial cells": 8
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize Xenium H&E images to 0.25 MPP and write paired image/label files."
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Root directory containing one folder per dataset id")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output root. Writes <save_dir>/<dataset_id>/{images,labels}")
    parser.add_argument("--dataset_ids", type=str, nargs="+", required=True,
                        help="Dataset folder names under --dataset_path")
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset_id in args.dataset_ids:
        process_dataset(args.dataset_path, args.save_dir, dataset_id)


def process_dataset(dataset_path, save_dir, dataset_id):
    print(f"Processing dataset: {dataset_id}...")
    
    # 1. Load and prepare adata
    adata = load_10x_Xenium_Anndata(dataset_path=os.path.join(dataset_path, dataset_id), dataset_id=dataset_id, anno_file=False)
    adata = adata[adata.obs['cell_type'] != 'Unknown'].copy()
    adata = adata[adata.obs['cell_type'] != 'Mixed'].copy()
    
    # Coordinates alignment
    adata_aligned = coords_alignment(adata, align=True)
    
    initial_cell_count = len(adata_aligned)
    adata_aligned = adata_aligned[~adata_aligned.obs['top'].isna()].copy()
    adata_aligned = adata_aligned[~adata_aligned.obs['bottom'].isna()].copy()
    adata_aligned = adata_aligned[~adata_aligned.obs['left'].isna()].copy()
    adata_aligned = adata_aligned[~adata_aligned.obs['right'].isna()].copy()
    filtered_cell_count = len(adata_aligned)
    print(f"Filtered out {initial_cell_count - filtered_cell_count} cells without boundary coordinates.")
    
    # 2. Load WSI, Rescale, and Pad
    image_path = adata_aligned.uns['spatialdata_attrs']['image_path']
    img_array = load_image(image_path)
    
    # Calculate scale factor and new dimensions
    scale_factor = 0.2125 / 0.25
    old_h, old_w = img_array.shape[:2]
    new_w = int(old_w * scale_factor)
    new_h = int(old_h * scale_factor)
    
    # Resize WSI image
    img_rescaled = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding to make dimensions divisible by 224
    pad_w = (224 - (new_w % 224)) % 224
    pad_h = (224 - (new_h % 224)) % 224
    
    # Determine white pad value depending on whether the image is RGB or Grayscale
    pad_value = [255] * img_rescaled.shape[2] if img_rescaled.ndim == 3 else 255
    
    # Apply padding to the right and bottom
    img_padded = cv2.copyMakeBorder(
        img_rescaled,
        top=0, bottom=pad_h, left=0, right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )
    
    # Save the rescaled and padded WSI
    save_dir_img = os.path.join(save_dir, dataset_id, "images")
    os.makedirs(save_dir_img, exist_ok=True)
    padded_image_path = os.path.join(save_dir_img, f"{dataset_id}.png")
    Image.fromarray(img_padded).save(padded_image_path)
    
    # 3. Rescale coordinates in adata_aligned
    adata_aligned.obsm['spatial_aligned'] = adata_aligned.obsm['spatial_aligned'] * scale_factor
    for col in ['top', 'bottom', 'left', 'right']:
        adata_aligned.obs[col] = adata_aligned.obs[col] * scale_factor
        
    # Update boundaries in adata.uns just in case it's needed later
    for key in['left', 'right', 'top', 'bottom']:
        adata_aligned.uns['boundaries'][key] *= scale_factor
        
    # 4. Prepare data for the MAT file
    num_cells = len(adata_aligned)
    
    # Generate ID mapping dictionaries
    id_array = np.arange(1, num_cells + 1, dtype=np.int32).reshape(-1, 1)
    barcode_to_id = {barcode: i + 1 for i, barcode in enumerate(adata_aligned.obs_names)}
    
    # 'class': mapping your 0-indexed dict to 1-indexed integers
    mapped_classes = adata_aligned.obs['cell_type'].map(cell_type_dict)
    # Any unmapped type will gracefully fall back to -1 before adding 1 (becomes 0, if desired)
    assert not mapped_classes.isna().any(), "mapped_classes contains NA values"
    class_array = (mapped_classes.astype(np.int32) + 1).values.reshape(-1, 1).astype(np.uint8)
    
    # 'bbox': format [ymin, ymax, xmin, xmax] matches [top, bottom, left, right]
    bbox_array = adata_aligned.obs[['top', 'bottom', 'left', 'right']].values.astype(np.int32)
    
    # 'centroid': [x, y]
    centroid_array = adata_aligned.obsm['spatial_aligned'].astype(np.float64)
    
    # 'expr': cell x gene expression matrix
    expr_array = adata_aligned.X.toarray() if sp.issparse(adata_aligned.X) else np.array(adata_aligned.X)
    
    # 5. Generate 'inst_map' by simulating the alignment on boundary polygons
    print(f"Generating instance map for {dataset_id}...")
    inst_map = np.zeros((img_padded.shape[0], img_padded.shape[1]), dtype=np.int32)
    
    # Read raw boundaries
    cell_boundaries_path = adata_aligned.uns['spatialdata_attrs']['nucleus_boundaries_path']
    bounds_df = pd.read_csv(cell_boundaries_path, compression='gzip', header=0)
    bounds_df['cell_id'] = bounds_df['cell_id'].astype(str)
    
    # Keep only those bounding coordinates of cells that survived QC
    bounds_df = bounds_df[bounds_df['cell_id'].isin(barcode_to_id)]
    
    # Compute alignment over polygon points
    align_matrix = load_transformation_matrix(adata_aligned.uns['spatialdata_attrs']['alignment_path'])
    raw_coords = bounds_df[['vertex_x', 'vertex_y']].values * (16 / 3.4)
    aligned_coords = project_points(align_matrix, raw_coords)
    
    # Scale polygon points to new WSI resolution
    scaled_coords = aligned_coords * scale_factor
    bounds_df['vertex_x_scaled'] = scaled_coords[:, 0]
    bounds_df['vertex_y_scaled'] = scaled_coords[:, 1]
    
    # Draw instances on the map
    for cell_id, group in bounds_df.groupby('cell_id'):
        poly = group[['vertex_x_scaled', 'vertex_y_scaled']].values.astype(np.int32)
        inst_id = barcode_to_id[cell_id]
        # Fill polygon using the numeric inst_id
        cv2.fillPoly(inst_map, [poly], int(inst_id))
        
    # 6. Save Mat File
    mat_dict = {
        'inst_map': inst_map,
        'id': id_array,
        'class': class_array,
        'bbox': bbox_array,
        'centroid': centroid_array,
        'expr': expr_array
    }
    
    save_dir_mat = os.path.join(save_dir, dataset_id, "labels")
    os.makedirs(save_dir_mat, exist_ok=True)
    mat_save_path = os.path.join(save_dir_mat, f"{dataset_id}.mat")
    sio.savemat(mat_save_path, mat_dict, do_compression=True, format='4')
    
    # 7. Generate TSV to pair cell numeric ID with cell barcodes
    tsv_path = os.path.join(save_dir_mat, f"{dataset_id}_cell_id_mapping.tsv")
    mapping_df = pd.DataFrame({
        'id': id_array.flatten(),
        'barcode': adata_aligned.obs_names
    })
    mapping_df.to_csv(tsv_path, sep='\t', index=False)

    print(f"Dataset {dataset_id} finalized and saved!")


if __name__ == "__main__":
    main()