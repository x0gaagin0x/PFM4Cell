import os
import yaml
import json
import shutil
import argparse
from torch import nn
from model import ColorCell
from safetensors.torch import load_file
from torch_geometric.loader import DataLoader
from data import load_split_dataset, GraphTileDataset
from utils import test, compute_classification_metrics

# Initialize the argument parser
parser = argparse.ArgumentParser()

# Add a command-line argument for the config file path
parser.add_argument("--config", type=str, required=True,
                    help="Path to the YAML configuration file")

# Parse the command-line arguments
args = parser.parse_args()

# Printing out the provided command-line parameters
print("Used config file: ", args.config)

# Load the YAML configuration file
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Load data
## suffix should be '_cells' when testing on NuCLS dataset.
(
 data_test_list,
 tile_test_list,
 slide_id_test_list,
 _,
 _,
 _,
 )  = load_split_dataset(dataset_train_path_list=[cfg['data']['dataset_path']], 
                         dataset_val_path_list=[],
                         graph_dirname=cfg['data']['graph_dirname'], 
                         image_dirname=cfg['data']['image_dirname'], suffix='')

test_set = GraphTileDataset(data_list=data_test_list, tile_list=tile_test_list, slide_id_list=slide_id_test_list,
                            cell_enc_type=cfg['model']['cell_enc_type'])

test_loader = DataLoader(test_set, batch_size=cfg['testing']['batch_size'], shuffle=False, 
                         num_workers=cfg['testing']['num_workers'], pin_memory=True, exclude_keys=['x_expr'])

# Create the save directory if it's not existed
save_directory = cfg['colorcell_ckpt']['save_dir']
os.makedirs(save_directory, exist_ok=True)
shutil.copy2(args.config, os.path.join(save_directory, 'test_config.yaml'))

# Set up the model
n_steps = cfg['model'].get('n_steps', 1)
gamma = cfg['model'].get('gamma', 1.5)
pre_mlp_hidden_dim = cfg['model'].get('pre_mlp_hidden_dim', 256)
pre_tabnet_hidden_dim = cfg['model'].get('pre_tabnet_hidden_dim', 32)
model = ColorCell(cell_enc_type=cfg['model']['cell_enc_type'], cell_emb_dim=cfg['model']['cell_emb_dim'],
                  mlp_hidden_dims=cfg['model']['mlp_hidden_dims'], output_dim=cfg['model']['output_dim'], 
                  norm_type=cfg['model']['norm_type'], act=nn.LeakyReLU(), cell_enc_dir=cfg['model']['cell_enc_dir'],
                  mode=cfg['model']['mode'], n_steps=n_steps, gamma=gamma, pre_mlp_hidden_dim=pre_mlp_hidden_dim, 
                  pre_tabnet_hidden_dim=pre_tabnet_hidden_dim)

# Load model weights from checkpoint
state_dict = load_file(os.path.join(save_directory, "model.safetensors"))
if cfg['model_mode'] == 'LP':
    model.classifier.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)

# Inference
test_results_dict = test(model=model, loader=test_loader, graph_type=cfg['model']['graph_type'],
                         mode=cfg['model']['mode'])

# Save test results
os.makedirs(os.path.join(save_directory, 'test_results'), exist_ok=True)
for slide_id, seg in test_results_dict.items():
    with open(os.path.join(save_directory, 'test_results', f'{slide_id}.json'), "w") as f:
        json.dump(seg, f, indent=4)

gt_directory = os.path.join(cfg['data']['dataset_path'], cfg['data']['gt_dirname'])
mat_suffix = cfg['data'].get('mat_suffix', '')

# compute metrics
(
 confusion_df, 
 f1_dict, 
 f1_joint_dict,
 auprc_dict, 
 auroc_dict, 
 accuracy,
 ) = compute_classification_metrics(gt_directory, test_results_dict, pairing_radius_um=cfg['testing']['pairing_radius_um'], 
                                    tile_res=cfg['data']['magnification'], gt_scale_factor=cfg['testing']['gt_scale_factor'],
                                    mat_suffix=mat_suffix)

# save metrics to files
label_map = {str(k): v for k, v in cfg['label_map'].items()}

## save confusion matrix
confusion_df = confusion_df.rename(
    index=lambda x: f"true_{label_map[x.split('_')[1]]}",
    columns=lambda x: f"pred_{label_map[x.split('_')[1]]}"
)

confusion_df.to_csv(os.path.join(save_directory, 'confusion_matrix.csv'), index=True)

## save f1, auprc, auroc, acc
f1_dict = {label_map[k]: v for k, v in f1_dict.items()}
f1_joint_dict = {label_map[k]: v for k, v in f1_joint_dict.items()}
auprc_dict = {label_map[k]: v for k, v in auprc_dict.items()}
auroc_dict = {label_map[k]: v for k, v in auroc_dict.items()}

scores_dict = {'f1': f1_dict, 'auprc': auprc_dict, 'auroc': auroc_dict, 'f1_joint': f1_joint_dict, 'acc': accuracy}

with open(os.path.join(save_directory, 'test_scores.json'), "w") as f:
    json.dump(scores_dict, f, indent=4)