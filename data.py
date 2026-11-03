import os
import torch
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def load_split_dataset(dataset_train_path_list, dataset_val_path_list, graph_dirname, image_dirname, suffix=''):
    """
    Load and split a multi-modal dataset (graph + image pairs) into training and validation subsets.

    This function assumes that each sample is represented by:
        1. A graph file (serialized with `torch.save`) located in `graph_dirname`.
        2. A corresponding image file (PNG format) located in `image_dirname`.

    The function performs the following steps:
        - Collects unique sample identifiers (slideID_tileID) across all dataset paths.
        - Randomly shuffles and splits the identifiers into training and validation sets
          according to the specified proportion.
        - Loads the graph objects and corresponding image tiles for each split.
        - Returns lists of graph data, image tiles, and their associated sample IDs.

    Parameters
    ----------
    dataset_path_list : list of str
        List of dataset root directories containing graph and image subfolders.
    graph_dirname : str
        Name of the subdirectory containing graph files.
    image_dirname : str
        Name of the subdirectory containing image files.
    train_pct : float
        Proportion of samples to allocate to the training set (0 < train_pct < 1).
    seed : int, optional (default=2434)
        Random seed for reproducible shuffling and splitting.

    Returns
    -------
    data_train_list : list
        List of loaded graph objects for the training set.
    tile_train_list : list
        List of corresponding image tiles (PIL.Image objects) for the training set.
    slide_id_train_list : list of str
        List of sample identifiers for the training set.
    data_val_list : list
        List of loaded graph objects for the validation set.
    tile_val_list : list
        List of corresponding image tiles (PIL.Image objects) for the validation set.
    slide_id_val_list : list of str
        List of sample identifiers for the validation set.
    """
     
    data_train_list, tile_train_list, slide_id_train_list = [], [], []
    data_val_list, tile_val_list, slide_id_val_list = [], [], []
    for dataset_path in dataset_train_path_list:
        graph_folder = os.path.join(dataset_path, graph_dirname)
        graph_filenames = os.listdir(graph_folder)
        graph_filenames = [f for f in graph_filenames if os.path.isfile(os.path.join(graph_folder, f))]
        
        for graph_filename in graph_filenames:
            sample_id = os.path.splitext(graph_filename)[0].rsplit("_", 1)[0]
            image_filename = os.path.splitext(graph_filename)[0].removesuffix(suffix) + ".png"
            data_train_list.append(torch.load(os.path.join(dataset_path, graph_dirname, graph_filename),
                                              weights_only=False))
            with Image.open(os.path.join(dataset_path, image_dirname, image_filename)) as img:
                tile_train_list.append(img.copy())
            slide_id_train_list.append(sample_id)
            
    for dataset_path in dataset_val_path_list:
        graph_folder = os.path.join(dataset_path, graph_dirname)
        graph_filenames = os.listdir(graph_folder)
        graph_filenames = [f for f in graph_filenames if os.path.isfile(os.path.join(graph_folder, f))]
        
        for graph_filename in graph_filenames:
            sample_id = os.path.splitext(graph_filename)[0].rsplit("_", 1)[0]
            image_filename = os.path.splitext(graph_filename)[0].removesuffix(suffix) + ".png"
            data_val_list.append(torch.load(os.path.join(dataset_path, graph_dirname, graph_filename),
                                              weights_only=False))
            with Image.open(os.path.join(dataset_path, image_dirname, image_filename)) as img:
                tile_val_list.append(img.copy())
            slide_id_val_list.append(sample_id)
    
    return data_train_list, tile_train_list,slide_id_train_list, data_val_list, tile_val_list, slide_id_val_list

def get_trans_func(cell_enc_type):
    if cell_enc_type == 'UNI' or cell_enc_type == 'UNI2-h' or cell_enc_type == 'Virchow' or cell_enc_type == 'Virchow2' or cell_enc_type == 'gigapath':
        tile_trasnform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
    elif cell_enc_type == 'HIPT':
        tile_trasnform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
    elif cell_enc_type == 'MUSK':
        tile_trasnform = transforms.Compose([
            transforms.Resize(384, interpolation=3, antialias=True),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
    elif cell_enc_type == 'CONCH':
        def _convert_to_rgb(image):
            return image.convert('RGB')
        
        tile_trasnform = transforms.Compose([
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
    
    return tile_trasnform

class GraphTileDataset(Dataset):
    def __init__(self, data_list, tile_list, slide_id_list, cell_enc_type):
        # Set transform function
        self.tile_trasnform = get_trans_func(cell_enc_type=cell_enc_type)
        
        # Load graphs and tiles
        self.data_list = data_list
        self.tile_list = tile_list
        self.slide_id_list = slide_id_list
        
    def __len__(self):
        return len(self.tile_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        tile = self.tile_trasnform(self.tile_list[idx])
        slide_id = self.slide_id_list[idx]
        
        return data, tile, slide_id

def extract_embeddings_and_update(dataset, teacher_model, use_source, batch_size, num_workers, device):
    """
    Performs offline feature extraction using the teacher model.
    Updates the dataset with new NODE-LEVEL features ('cell_expr_fused' and 'teacher_pred').
    
    Args:
        dataset: GraphTileDataset (CPU).
        teacher_model: Pre-trained model.
        use_source: Control flag/tensor.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        device: Device to run the teacher model on.
        
    Returns:
        List[Data]: A list of Data objects on CPU, strictly preserving the original order,
                    with new node-level attributes attached.
    """
    
    # 1. Initialize DataLoader
    # IMPORTANT: shuffle=False ensures the output list matches the input list order (slide_id_list, etc.)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    updated_data_list = []
    
    with torch.no_grad():
        iterator = tqdm(loader, desc="Extracting Teacher Features")
        
        for batch in iterator:
            # 2. Move data to GPU
            data_batch = batch[0].to(device)
            tile_batch = batch[1].to(device)
            
            # 3. Forward Pass
            # Expected outputs:
            # cell_expr_fused: [Total_Nodes_In_Batch, Dim1]
            # teacher_pred:    [Total_Nodes_In_Batch, Dim2] (As confirmed by user: Node-Level)
            cell_expr_fused, _, _, _, teacher_pred = teacher_model(tile_batch, data_batch, use_source)
            
            # 4. De-batching (Split the Super-Graph back into individual graphs)
            # This reconstructs the list of Data objects with standard attributes (x, edge_index, etc.)
            # However, it DOES NOT automatically split our new variables `teacher_pred` or `cell_expr_fused`
            data_list_from_batch = data_batch.to_data_list()
            
            # 5. Manually slice and assign the new node features
            start_node_idx = 0
            
            for graph_obj in data_list_from_batch:
                num_nodes = graph_obj.num_nodes
                
                # Define the slice range for the current graph
                end_node_idx = start_node_idx + num_nodes
                
                # --- Assign Node-Level Features ---
                
                # Slice 'cell_expr_fused' for the current graph
                # Shape: [num_nodes, Dim1]
                graph_obj.cell_expr_fused = cell_expr_fused[start_node_idx : end_node_idx]
                
                # Slice 'teacher_pred' for the current graph
                # Shape: [num_nodes, Dim2]
                graph_obj.teacher_pred = teacher_pred[start_node_idx : end_node_idx]
                
                # Update the pointer for the next graph in the batch
                start_node_idx = end_node_idx
                
                # 6. Offload to CPU
                # Move the object (and its new attributes) to CPU to save GPU memory
                graph_obj = graph_obj.to('cpu')
                
                updated_data_list.append(graph_obj)
                
    return updated_data_list