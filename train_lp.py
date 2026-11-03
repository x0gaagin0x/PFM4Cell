import os
import yaml
import json
import torch
import shutil
import argparse
from torch import nn
from accelerate import Accelerator
from encoder import Encoder
from torch_geometric.loader import DataLoader
from data import load_split_dataset, GraphTileDataset
from utils import set_seed, seed_worker
from lp_utils import train_one_epoch, evaluate
from tqdm import tqdm
from model import assign_patch_embeddings_to_instances
from torch_geometric.data import Batch
from classifier import Classifier
from data import get_trans_func

# Initialize the argument parser
parser = argparse.ArgumentParser()

# Add a command-line argument for the config file path
parser.add_argument("--config", type=str, required=True,
                    help="Path to the YAML configuration file")

parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

# Parse the command-line arguments
args = parser.parse_args()

# Printing out the provided command-line parameters
print("Used config file: ", args.config)

# Load the YAML configuration file
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Set random seed
set_seed(seed=cfg['random_seed'])

# Set up the accelerator
accelerator = Accelerator(gradient_accumulation_steps=cfg['lp_training']['accumulation_steps'])

## Load data
(
 data_train_list,
 tile_train_list,
 slide_id_train_list,
 data_val_list,
 tile_val_list,
 slide_id_val_list,
 )  = load_split_dataset(dataset_train_path_list=cfg['data']['dataset_train_path_list'], 
                         dataset_val_path_list=cfg['data']['dataset_val_path_list'],
                         graph_dirname=cfg['data']['graph_dirname'], 
                         image_dirname=cfg['data']['image_dirname'],
                         suffix='')

# Load pretrained encoder
cell_enc = Encoder(enc_type=cfg['model']['cell_enc_type'], model_dir=cfg['model']['cell_enc_dir'])

## generate cell embeddings from pretrained model
tile_transform = get_trans_func(cell_enc_type=cfg['model']['cell_enc_type'])

cell_enc.to(accelerator.device)
cell_enc.eval()

if cfg['model']['cell_enc_type'] == 'MUSK':
    patch_size = 224/24
elif cfg['model']['cell_enc_type'] == 'CONCH' or cfg['model']['cell_enc_type'] == 'UNI' or cfg['model']['cell_enc_type'] == 'gigapath':
    patch_size = 16
else:
    patch_size = 14
    
with torch.no_grad():
    #  Batch.from_data_list([data_train_list[i]]).to(accelerator.device)
    for i in tqdm(range(len(data_train_list))):
        cls_emb, patch_emb = cell_enc(tile_transform(tile_train_list[i]).unsqueeze(dim=0).to(accelerator.device))
        cell_emb = assign_patch_embeddings_to_instances(patch_emb, 
                                                        Batch.from_data_list([data_train_list[i]]).to(accelerator.device),
                                                        patch_size=patch_size)
        data_train_list[i].cell_emb = cell_emb.detach().cpu()
        data_train_list[i].cls_emb = cls_emb.detach().cpu()
    
    for i in tqdm(range(len(data_val_list))):
        cls_emb, patch_emb = cell_enc(tile_transform(tile_val_list[i]).unsqueeze(dim=0).to(accelerator.device))
        cell_emb = assign_patch_embeddings_to_instances(patch_emb, 
                                                        Batch.from_data_list([data_val_list[i]]).to(accelerator.device),
                                                        patch_size=patch_size)
        data_val_list[i].cell_emb = cell_emb.detach().cpu()
        data_val_list[i].cls_emb = cls_emb.detach().cpu()

# Create dataset
train_set = GraphTileDataset(data_list=data_train_list, tile_list=tile_train_list, slide_id_list=slide_id_train_list, 
                             cell_enc_type=cfg['model']['cell_enc_type'])
val_set = GraphTileDataset(data_list=data_val_list, tile_list=tile_val_list, slide_id_list=slide_id_val_list, 
                           cell_enc_type=cfg['model']['cell_enc_type'])

# Create DataLoader for iteration
train_loader = DataLoader(train_set, batch_size=cfg['lp_training']['batch_size'], shuffle=True, 
                          num_workers=cfg['lp_training']['num_workers'], worker_init_fn=seed_worker, 
                          generator=torch.Generator().manual_seed(cfg['random_seed']), pin_memory=True, exclude_keys=['x_expr'])

val_loader = DataLoader(val_set, batch_size=cfg['lp_training']['batch_size'], shuffle=False, 
                        num_workers=cfg['lp_training']['num_workers'], pin_memory=True, exclude_keys=['x_expr'])

# Set up the model
n_steps = cfg['model'].get('n_steps', 1)
gamma = cfg['model'].get('gamma', 1.5)
sparsity_loss_weight = cfg['lp_training'].get('sparsity_loss_weight', 0)
pre_mlp_hidden_dim = cfg['model'].get('pre_mlp_hidden_dim', 256)
pre_tabnet_hidden_dim = cfg['model'].get('pre_tabnet_hidden_dim', 32)
model = Classifier(emb_dim=cfg['model']['cell_emb_dim'], mlp_hidden_dims=cfg['model']['mlp_hidden_dims'], 
                   output_dim=cfg['model']['output_dim'], norm_type=cfg['model']['norm_type'], act=nn.LeakyReLU(),
                   mode=cfg['model']['mode'], n_steps=n_steps, gamma=gamma,
                   pre_mlp_hidden_dim=pre_mlp_hidden_dim, pre_tabnet_hidden_dim=pre_tabnet_hidden_dim)

# Set up the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lp_training']['learning_rate'], 
                              weight_decay=cfg['lp_training']['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                       T_max=cfg['lp_training']['n_epoch'], 
                                                       eta_min=cfg['lp_training']['learning_rate']/10)

# Wrap model, optimizer and loader by accelerator
model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, 
                                                                train_loader)
val_loader = accelerator.prepare(val_loader)

# Create the save directory if it's not existed
save_directory = cfg['lp_ckpt']['save_dir']
os.makedirs(save_directory, exist_ok=True)
shutil.copy2(args.config, os.path.join(save_directory, 'train_config_lp.yaml'))

# ------- Training Loop -------
# Early stopping parameters
class_weights = cfg['lp_training']['class_weights'] if cfg['lp_training']['use_weighted_loss'] else None
wait = 0
best_val_avg_auprc = float("-inf")
best_epoch = 0
val_results_dict = {}

for epoch in range(1, cfg['lp_training']['n_epoch'] + 1):
    train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, accelerator=accelerator, 
                                 loss_type=cfg['lp_training']['loss_type'], num_classes=cfg['model']['output_dim'], 
                                 class_weights=class_weights, graph_type=cfg['model']['graph_type'],
                                 mode=cfg['model']['mode'], sparsity_loss_weight=sparsity_loss_weight)
    
    val_loss, val_acc, val_macro_f1, val_avg_auprc, val_avg_auroc = evaluate(model=model, loader=val_loader, 
                                                                             accelerator=accelerator, 
                                                                             loss_type=cfg['lp_training']['loss_type'], 
                                                                             num_classes=cfg['model']['output_dim'], 
                                                                             class_weights=class_weights,
                                                                             graph_type=cfg['model']['graph_type'],
                                                                             mode=cfg['model']['mode'],
                                                                             sparsity_loss_weight=sparsity_loss_weight)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Epoch {epoch:03d} | Val Macro-F1: {val_macro_f1:.4f} | Val Avg AUPRC: {val_avg_auprc:.4f} | Val Avg AUROC: {val_avg_auroc:.4f}")
        
        val_result = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
                      "val_macro_f1": val_macro_f1, "val_avg_auprc": val_avg_auprc, "val_avg_auroc": val_avg_auroc,}
        
        val_results_dict[f'epoch_{epoch}'] = val_result
        
        # Check for improvement and save best model
        if val_avg_auprc > best_val_avg_auprc:
            best_val_avg_auprc = val_avg_auprc
            best_epoch = epoch
            accelerator.save_model(model, save_directory)
            print(f"✅ New best model saved at epoch {epoch}")

# Save val results
with open(os.path.join(save_directory, 'val_results_lp.json'), "w") as f:
    json.dump(val_results_dict, f, indent=4)
    
print(f"Best validation Macro F1: {best_val_avg_auprc:.4f} at epoch {best_epoch}")