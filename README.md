# PFM4Cell

## Installation

1. Clone repository:

```shell
git clone https://github.com/x0gaagin0x/PFM4Cell.git
```

2. Create virtual environment:

```shell
conda create --name PFM4Cell python=3.11
```

3. Activate virtual environment:

```shell
conda activate PFM4Cell
```

4. Install dependencies:

```shell
cd PFM4Cell
pip install -r requirements.txt
```

## Data preprocessing

The Spatial Pan-cancer Atlas of Cells (SPAC) dataset can be downloaded from [link](https://drive.google.com/file/d/16h-zTr3x_APV351kHP5I5d-G5nSB7-cA/view?usp=sharing). Links to the other required files are provided in `link.md`.

Required files for each Xenium dataset:

1. a h5ad file with cell type annotation: `Xenium_data_annotated.h5ad`
2. a nucleus boundaries file: `nucleus_boundaries.csv.gz`
3. a H&E whole slide image paired with spatial transcriptomics data: `*_he_image.ome.tif`
4. an image alignment file: `*_he_imagealignment.csv`

Preprocessing:

1. Resize images and rescale coordinates to a resolution of 0.25 MPP, then generate the paired .mat files:

```shell
python ./preprocessing/Data_preparation.py --dataset_path /path/to/xenium --save_dir /path/to/output --dataset_ids <dataset_id>
```

2. Split images into non-overlapped 224x224 patches and map cells to their corresponding patches:

```shell
python ./preprocessing/mat_split_spatch.py --base_dir /path/to/output/<dataset_id>
```

## Model training

Before running, please replace the paths in config files with your own.

```shell
model="UNI"
fold="1"
# Linear Probing
accelerate launch --num_processes=1 train_lp.py --config logs/Benchmark_cell_pred_lp/configs/${model}/config_train_${fold}.yaml

# Linear Probing then Fine-tuning
accelerate launch --num_processes=1 train_lp.py --config logs/Benchmark_cell_pred_lpft/configs/${model}/config_train_${fold}.yaml && accelerate launch --num_processes=1 train_ft.py --config logs/Benchmark_cell_pred_lpft/configs/${model}/config_train_${fold}.yaml
```

## Quick inference with the cell classifier

1. Install CellViT++ following the github page: https://github.com/TIO-IKIM/CellViT-plus-plus/tree/main
2. Perform cell segmentation on WSI via CellViT++:

```shell
wsi_mpp=0.25
wsi_magnification=40

cellvit-inference \
    --model SAM \
    --nuclei_taxonomy binary \
    --outdir "/path/to/output" \
    --memory 10240 \
    --ray_worker 2 \
    process_wsi \
    --wsi_path "/path/to/image" \
    --wsi_mpp ${wsi_mpp} \
    --wsi_magnification ${wsi_magnification}
```

3. Use the output `cell_detection.json` for cell type inference:

```shell
python cell_type_inference.py --wsi_path "/path/to/image" --cell_seg_path "/path/to/cell_detection.json" --slide_mpp ${wsi_mpp} --configs inference_config/cell_type_inference_config.yaml --outdir "/path/to/output"
```
