# TASAD: Two‑Stage Coarse‑to‑Fine Anomaly Segmentation & Detection

This repository provides the code for **TASAD**, a two‑stage coarse‑to‑fine model for **industrial image anomaly detection and segmentation**, along with **SAIM** (Superpixel‑based Anomaly Insertion Method) for generating pseudo‑anomalies of different sizes to improve generalization.

---

## Overview

### TASAD workflow
![fasad_model](https://github.com/RizwanAliQau/tasad/assets/29249233/7090cb30-663a-4e6a-ae27-a35b2f65793e)

### SAIM (Superpixel‑based Anomaly Insertion Method)
**SAIM** generates pseudo‑anomalies in **three sizes (small / medium / large)** by controlling the number of superpixel segments taken from the anomaly source image.

#### Pseudo‑anomaly insertion with SAIM
![saim-1](https://github.com/RizwanAliQau/tasad/assets/29249233/88ffd8aa-ed87-4da0-9e0b-cf55e4f80b4c)

#### Example: controlling anomaly sizes via superpixel segments
![Size_of_anom-1](https://github.com/RizwanAliQau/tasad/assets/29249233/371e2bf9-9c8a-44d7-98e3-8e8823fd4b71)

---

## Project structure (required folders)

Create the following directories in the repository root before training/testing:

```text
├── best_weights_model_1
├── best_weights_model_2
├── checkpoints
├── data
├── logs
├── test_weights
└── weights
```

---

## Setup

### 1) Create a conda environment

```bash
conda create -n ENVNAME --file requirement.txt
conda activate ENVNAME
pip install -r requirement_pip.txt
```

> Notes:
> - `requirement.txt` is used for conda packages.
> - `requirement_pip.txt` is used for pip packages.

---

## Pretrained weights (for testing)

1. Download weights from Google Drive:
   - https://drive.google.com/drive/folders/10Z0MNGY9codk0F-h59roTUr4Xeay2IPO?usp=share_link
2. Place the downloaded files into the appropriate folder (commonly `./weights/` or `./test_weights/` depending on your usage).

---

## Testing

### TASAD testing command (MVTec)

```bash
python ./main/test_seg_model.py \
  --gpu_id 0 \
  --model_name cas_seg_model_weights_mvtech_ \
  --data_path ./data/ \
  --checkpoint_path ./weights/ \
  --both_model 1 \
  --obj_list_all carpet,grid,leather,tile,wood,bottle,capsule,pill,transistor,zipper,cable,hazelnut,metal_nut,screw,toothbrush
```

**Arguments (high level):**
- `--both_model 1` runs both stages (coarse + fine) if supported by the script.
- `--obj_list_all` is the list of MVTec categories to test.

---

## Training

### 1) Download datasets

- **MVTec AD dataset** (put under `./data/` as expected by the code):
  - https://www.mvtec.com/company/research/datasets/mvtec-ad
- **DTD texture dataset** (used as anomaly source images; place under the path you pass via `--anomaly_source_path`):
  - https://www.robots.ox.ac.uk/~vgg/data/dtd/

> Ensure your dataset folders match what the training scripts expect.

---

### 2) CAS training (coarse stage)

```bash
python ./main/cas_train.py \
  --gpu_id 0 \
  --gpu_id_validation 0 \
  --obj_id -1 \
  --lr 0.0001 \
  --bs 1 \
  --epochs 4000 \
  --data_path ./data/ \
  --anomaly_source_path ./anomlay_addition_data/ \
  --checkpoint_path ./test_weights/ \
  --log_path ./logs/ \
  --checkpoint_cas_model "" \
  --visualize True \
  --class_name hazelnut \
  --best_model_save_path ./best_weights_model_1/
```

---

### 3) FAS training (fine stage)

```bash
python ./main/fas_train.py \
  --train_gpu_id 0 \
  --val_gpu_id 0 \
  --obj_id -1 \
  --lr 0.0001 \
  --bs 1 \
  --epochs 4000 \
  --data_path ./data/ \
  --anomaly_source_path ./anomlay_addition_data/ \
  --cas_model_path ./test_weights/ \
  --checkpoint_path ./checkpoints/ \
  --log_path ./logs/ \
  --checkpoint_cas_weights ./test_weights/cas_seg_model_weights_mvtech_ \
  --checkpoint_fas_weights ./test_weights/fas_seg_model_weights_mvtech_ \
  --visualize True \
  --class_name hazelnut \
  --datatype png
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{shah2023two,
  title={Two-stage coarse-to-fine image anomaly segmentation and detection model},
  author={Shah, Rizwan Ali and Urmonov, Odilbek and Kim, HyungWon},
  journal={Image and Vision Computing},
  volume={139},
  pages={104817},
  year={2023},
  publisher={Elsevier}
}
```

---

## Acknowledgements

- **DRAEM** (code and model weights):
  https://github.com/VitjanZ/DRAEM
