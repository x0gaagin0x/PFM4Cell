import torch.nn as nn
from encoder import Encoder
from classifier import Classifier, MLP
import torch
from torch_geometric.data import Data

def assign_patch_embeddings_to_instances(
    patch_embeddings: torch.Tensor,
    data: Data,
    patch_size: int = 14,
    img_size: int = 224
) -> torch.Tensor:
    """
    Assigns patch embeddings to instances based on their bounding boxes.

    For each instance, it identifies all the ViT patches that overlap with its
    bounding box and computes the average of their embeddings.

    Args:
        patch_embeddings (torch.Tensor): The embeddings for all patches from the ViT.
                                         Shape: [batch_size, num_patches, D].
        data (torch_geometric.data.Data): The batched data object from PyG DataLoader.
                                          It must contain:
                                          - data.x_bbox: Bounding boxes for each instance (node).
                                            Shape: [S, 4], Format: [ymin, ymax, xmin, xmax].
                                          - data.batch: A tensor mapping each node to its
                                            graph/image index in the batch. Shape: [S].
                                          - data.patch_center: The center coordinates of the
                                            224x224 crop for each image in the batch.
                                            Shape: [batch_size, 2], Format: [x_center, y_center].
        patch_size (int, optional): The size of a single patch (e.g., 14 for ViT-P/14). Defaults to 14.
        img_size (int, optional): The input image size for the ViT. Defaults to 224.

    Returns:
        torch.Tensor: The computed instance embeddings. Shape: [S, D].
    """
    # Ensure patch_center has the correct shape for batch processing
    if not hasattr(data, 'patch_center') or data.patch_center.dim() == 1:
        raise ValueError("data.patch_center must be a [batch_size, 2] tensor.")

    embedding_dim = patch_embeddings.shape[-1]
    device = patch_embeddings.device
    grid_size = img_size // patch_size
    
    # Get the correct patch center for each instance using the batch index
    # data.patch_center is [batch_size, 2], data.batch is [S]
    # instance_centers becomes [S, 2]
    instance_centers = data.patch_center[data.batch]

    # Calculate the top-left corner of the 224x224 crop for each instance
    # Shape: [S, 2]
    crop_top_left = instance_centers - (img_size // 2)

    # Convert global bbox coordinates to local coordinates within the 224x224 crop
    # data.x_bbox is [S, 4] -> [ymin, ymax, xmin, xmax]
    # crop_top_left is [S, 2] -> [x, y]
    # We need to subtract [y_tl, y_tl, x_tl, x_tl]
    local_bboxes = data.x_bbox - torch.cat(
        [crop_top_left[:, [1]], crop_top_left[:, [1]], crop_top_left[:, [0]], crop_top_left[:, [0]]],
        dim=1
    )

    # Clip the coordinates to be within the image bounds [0, img_size - 1]
    local_bboxes_clipped = torch.clamp(local_bboxes, 0, img_size - 1)

    # Convert pixel coordinates to patch indices
    # We use floor division to find which patch index a coordinate falls into
    start_patch_coords = torch.floor(local_bboxes_clipped[:, [2, 0]] / patch_size).long() # xmin, ymin -> col, row
    end_patch_coords = torch.floor(local_bboxes_clipped[:, [3, 1]] / patch_size).long()   # xmax, ymax -> col, row

    instance_embeddings_list = []
    
    # Loop through each instance to gather and average embeddings
    for i in range(data.num_nodes):
        batch_idx = data.batch[i]
        
        # Get the range of patch indices (row and column) for the current instance
        start_col, start_row = start_patch_coords[i]
        end_col, end_row = end_patch_coords[i]

        # If the clipped box has zero area in terms of patches, we can't find any.
        # This can happen if the entire bbox is outside the crop.
        if start_row > end_row or start_col > end_col:
            # Assign a zero vector if no patches are covered
            instance_embeddings_list.append(torch.zeros(embedding_dim, device=device))
            continue

        # Create a grid of indices for the covered patches
        # Example: if start_row=1, end_row=2, start_col=3, end_col=4
        # covered_rows will be [1, 1, 2, 2]
        # covered_cols will be [3, 4, 3, 4]
        covered_rows = torch.arange(start_row, end_row + 1, device=device)
        covered_cols = torch.arange(start_col, end_col + 1, device=device)
        
        # This creates all combinations of row and col indices
        mesh_rows, mesh_cols = torch.meshgrid(covered_rows, covered_cols, indexing='ij')
        
        # Convert 2D patch coordinates (row, col) to 1D sequence index
        # The patch embeddings are flattened in row-major order
        flat_indices = (mesh_rows.flatten() * grid_size + mesh_cols.flatten()).long()
        
        # Select the patch embeddings for the current image
        image_patches = patch_embeddings[batch_idx]
        
        # Gather the embeddings for all covered patches
        relevant_embeddings = image_patches[flat_indices]
        
        # Compute the mean to get the final instance embedding
        instance_embedding = relevant_embeddings.mean(dim=0)
        instance_embeddings_list.append(instance_embedding)

    # Stack all instance embeddings into a single tensor
    return torch.stack(instance_embeddings_list)

class ColorCell(nn.Module):
    def __init__(self, cell_enc_type, cell_emb_dim, mlp_hidden_dims, output_dim, 
                 norm_type='layer', act=nn.LeakyReLU(), cell_enc_dir=None, mode='MLP', 
                 n_steps=1, gamma=1.5, pre_mlp_hidden_dim=256, pre_tabnet_hidden_dim=32):
        super().__init__()
        
        self.enc_type = cell_enc_type
        self.enc = Encoder(enc_type=cell_enc_type, model_dir=cell_enc_dir)
        
        self.classifier = Classifier(emb_dim=cell_emb_dim, mlp_hidden_dims=mlp_hidden_dims, output_dim=output_dim,
                                     norm_type=norm_type, act=act, mode=mode, n_steps=n_steps, gamma=gamma,
                                     pre_mlp_hidden_dim=pre_mlp_hidden_dim, 
                                     pre_tabnet_hidden_dim=pre_tabnet_hidden_dim)
        
    def forward(self, tile, data):
        
        cls_emb, patch_emb = self.enc(tile)
        
        if self.enc_type == 'MUSK':
            cell_emb = assign_patch_embeddings_to_instances(patch_emb, data, patch_size=224/24)
        elif self.enc_type == 'UNI' or self.enc_type == 'CONCH' or self.enc_type == 'gigapath':
            cell_emb = assign_patch_embeddings_to_instances(patch_emb, data, patch_size=16)
        else:
            cell_emb = assign_patch_embeddings_to_instances(patch_emb, data, patch_size=14)
        
        if self.classifier.mode == 'MLP':
            pred = self.classifier(cell_emb)
            return pred

        elif self.classifier.mode == 'both':
            pred, loss_masks = self.classifier(cell_emb, data.x) # data.x: manual extracted features
            return pred, loss_masks

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.nn.functional.softplus(x), min=1e-4, max=1e4)

class GenePredictor(nn.Module):
    def __init__(self, cell_enc_type, cell_emb_dim, expr_dim, mlp_hidden_dims, norm_type='layer', act=nn.LeakyReLU(),
                 cell_enc_dir=None):
        super().__init__()
        
        self.enc_type = cell_enc_type
        self.enc = Encoder(enc_type=cell_enc_type, model_dir=cell_enc_dir)
        
        self.predictor = Predictor(cell_emb_dim=cell_emb_dim, expr_dim=expr_dim, mlp_hidden_dims=mlp_hidden_dims, 
                                   norm_type=norm_type, act=act)
        
    def forward(self, tile, data):
        cls_emb, patch_emb = self.enc(tile)
        
        if self.enc_type == 'MUSK':
            cell_emb = assign_patch_embeddings_to_instances(patch_emb, data, patch_size=224/24)
        elif self.enc_type == 'UNI' or self.enc_type == 'CONCH' or self.enc_type == 'gigapath':
            cell_emb = assign_patch_embeddings_to_instances(patch_emb, data, patch_size=16)
        else:
            cell_emb = assign_patch_embeddings_to_instances(patch_emb, data, patch_size=14)
        
        expr_pred, expr_disp_pred = self.predictor(cell_emb)
        
        return expr_pred, expr_disp_pred
    
class Predictor(nn.Module):
    def __init__(self, cell_emb_dim, expr_dim, mlp_hidden_dims, norm_type='layer', act=nn.LeakyReLU()):
        super().__init__()
        
        self.expr_predictor = MLP(input_dim=cell_emb_dim, hidden_dims=mlp_hidden_dims, 
                                  output_dim=expr_dim, norm_type=norm_type, act=act)
        self.expr_disp_predictor = MLP(input_dim=cell_emb_dim, hidden_dims=mlp_hidden_dims, 
                                       output_dim=expr_dim, norm_type=norm_type, act=act)
        self.MeanAct = MeanAct()
        self.DispAct = DispAct()
    
    def forward(self, cell_emb):
        expr_pred = self.expr_predictor(cell_emb)
        expr_pred = self.MeanAct(expr_pred)
        
        expr_disp_pred = self.expr_disp_predictor(cell_emb)
        expr_disp_pred = self.DispAct(expr_disp_pred)
        
        return expr_pred, expr_disp_pred