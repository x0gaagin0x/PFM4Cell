# Load packages
import numpy as np
import pandas as pd

def load_transformation_matrix(alignment_file_path):
    """
    Loads the transformation matrix from a file.

    Parameters:
    - alignment_file_path: Path to the alignment file containing the transformation matrix.

    Returns:
    - transformation_matrix: A 2x3 NumPy array representing the affine transformation matrix.
    """
    # Example: Assuming the alignment file contains the transformation matrix in a simple text format
    # You may need to adjust this based on the actual format of your alignment file
    with open(alignment_file_path, 'r') as file:
        lines = file.readlines()
        # Example format: Each line contains a row of the transformation matrix
        # For a 2x3 matrix, you would have two lines with three values each
        matrix_rows = [list(map(float, line.strip().split(','))) for line in lines]
        transformation_matrix = np.array(matrix_rows)
    
    return transformation_matrix

def project_points(transform_matrix, cell_coords):
    """
    Project coordinates from aligned image space to original unaligned space.
    
    Args:
        transform_matrix: 3x3 affine transformation matrix (aligned = M @ original)
        cell_coords: Nx2 numpy array of (x, y) coordinates in unaligned space
    
    Returns:
        Nx2 numpy array of aligned cell coordinates
    """
    # Validate input shape
    if cell_coords.shape[1] != 2:
        raise ValueError("Input points must be in Nx2 format")

    # Add homogeneous coordinates (column of ones)
    ones = np.ones((cell_coords.shape[0], 1))
    cells_homog = np.hstack([cell_coords, ones]).T  # 3xN matrix

    # Calculate inverse transformation
    M_inv = np.linalg.inv(transform_matrix)

    # Apply inverse transformation
    transformed_homog = M_inv @ cells_homog

    # Convert back to Cartesian coordinates
    aligned_cell_coords = (transformed_homog[:2] / transformed_homog[2]).T

    return aligned_cell_coords

# Function incorporating all steps in alignment
def coords_alignment(adata, align=True):
    # Set file paths
    alignment_file_path = adata.uns['spatialdata_attrs']['alignment_path']
    cell_boundaries_file_path = adata.uns['spatialdata_attrs']['nucleus_boundaries_path']
    
    # Load cell boundaries file
    cell_boundaries_df = pd.read_csv(cell_boundaries_file_path, compression='gzip', header=0)
    
    # Load transformation matrix
    if align:
        transformation_matrix = load_transformation_matrix(alignment_file_path)
    
    # convert coordinates unit from micrometer to pixel
    adata.obsm['spatial'] = adata.obsm['spatial']*16/3.4
    cell_boundaries = np.array(cell_boundaries_df[['vertex_x', 'vertex_y']]*16/3.4)
    
    # croodinates alignment based on the transformation matrix
    if align:
        adata.obsm['spatial_aligned'] = project_points(transformation_matrix, adata.obsm['spatial'])
        cell_boundaries_aligned = project_points(transformation_matrix, cell_boundaries)
        cell_boundaries_df_aligned = cell_boundaries_df
        cell_boundaries_df_aligned[['vertex_x', 'vertex_y']] = pd.DataFrame(cell_boundaries_aligned, 
                                                                            index=cell_boundaries_df_aligned.index)
    else:
        adata.obsm['spatial_aligned'] = adata.obsm['spatial']
        cell_boundaries_aligned = cell_boundaries_df[['vertex_x', 'vertex_y']].to_numpy()
        cell_boundaries_df_aligned = cell_boundaries_df
        
    
    # Group by cell_id and calculate boundaries.
    cell_boundaries_df_aligned = cell_boundaries_df_aligned.groupby("cell_id").agg(
        left=('vertex_x', 'min'),
        right=('vertex_x', 'max'),
        top=('vertex_y', 'min'),    # if using typical image coordinates; swap to 'max' if y grows upward
        bottom=('vertex_y', 'max')  # likewise, swap to 'min' if necessary
    )
    
    cell_boundaries_df_aligned.index = cell_boundaries_df_aligned.index.astype(str)
    
    adata.obs = adata.obs.join(cell_boundaries_df_aligned, how='left')
    
    # Expand boundaies to capture cell surroundings
    # adata.obs['left'] = adata.obs['left'] - (adata.obs['right'] - adata.obs['left'])*0.25
    # adata.obs['right'] = adata.obs['right'] + (adata.obs['right'] - adata.obs['left'])*0.25
    # adata.obs['top'] = adata.obs['top'] - (adata.obs['bottom'] - adata.obs['top'])*0.25
    # adata.obs['bottom'] = adata.obs['bottom'] + (adata.obs['bottom'] - adata.obs['top'])*0.25
    
    # Determin the boundaries of the spatial data
    left = np.min(cell_boundaries_aligned[:, 0])
    right = np.max(cell_boundaries_aligned[:, 0])
    top = np.min(cell_boundaries_aligned[:, 1])
    bottom = np.max(cell_boundaries_aligned[:, 1])
    
    # Save the boundaries in adata.uns
    adata.uns['boundaries'] = {'left': left, 'right': right,
                               'top': top, 'bottom': bottom}
        
    return adata
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        