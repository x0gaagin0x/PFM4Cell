# Functions for loading various scRNA-seq and spatial transcriptomics datasets

# Load packages
import os
import pathlib
import scanpy as sc
import pandas as pd
import tifffile
from PIL import Image
import numpy as np

def find_file_path(directory, filename_suffix):
    """Find the path of a specific file in a directory."""
    directory_path = pathlib.Path(directory)
    for file in directory_path.glob(f"*{filename_suffix}"):
        if file.is_file():
            return file

def load_10x_Xenium_Anndata(dataset_path, dataset_id, anno_file=False): # 0.2125 um/px
    adata_path = os.path.join(dataset_path, "Xenium_data_annotated.h5ad")
    image_path = find_file_path(dataset_path, 'he_image.ome.tif')
    alignment_path = find_file_path(dataset_path, 'he_imagealignment.csv')
    cell_boundaries_path = os.path.join(dataset_path, 'cell_boundaries.csv.gz')
    nucleus_boundaries_path = os.path.join(dataset_path, 'nucleus_boundaries.csv.gz')
    
    adata = sc.read_h5ad(adata_path)
    adata.obs['Sample'] = dataset_id
    adata.uns['spatialdata_attrs']['image_path'] = image_path
    adata.uns['spatialdata_attrs']['alignment_path'] = alignment_path
    adata.uns['spatialdata_attrs']['cell_boundaries_path'] = cell_boundaries_path
    adata.uns['spatialdata_attrs']['nucleus_boundaries_path'] = nucleus_boundaries_path
    
    if anno_file:
        anno_path = find_file_path(os.path.join(dataset_path, dataset_id), 'cell_groups.csv')
        anno_info = pd.read_csv(anno_path, header=0)
        anno_info.set_index('cell_id', inplace=True)
        adata.obs['cell_type'] = adata.obs_names.map(anno_info['group'])
    
    return adata

def load_image(image_path):
    """
    Loads an image from the given file path and returns it as a NumPy array.
    
    Supported image formats:
      - JPG, PNG, and TIFF (non-OME)
      - OME-TIFF format if the filename ends with 'ome.tif' or 'ome.tiff'
    
    Parameters:
        image_path (str): The file path to the image.
        
    Returns:
        np.ndarray: The image represented as a NumPy array.
        
    Raises:
        IOError: If the image fails to load.
    """
    try:
        file_name = os.path.basename(image_path).lower()
        
        # Check for OME-TIFF by checking file ending
        if file_name.endswith("ome.tif") or file_name.endswith("ome.tiff"):
            with tifffile.TiffFile(image_path) as tif:
                image_array = tif.pages[0].asarray()
        else:
            with Image.open(image_path) as img:
                image_array = np.array(img)
        
        return image_array
    except Exception as e:
        raise IOError(f"Unable to load image from {image_path}. Error: {e}")