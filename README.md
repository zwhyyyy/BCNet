# BCNet

DINOv3 ViT-H+/16 backbone with LoRA fine-tuning for AI-generated image detection.

This repository contains the implementation of **Revisiting Deepfake Detection:
BCNet for Robust Generalization Beyond Semantic Dependence**, accepted by ECCV
2026.

For questions, please contact 25125405@bjtu.edu.cn.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the DINOv3 pretrained weights separately and set the local weight
directory in `models/dino_model.py`:

```python
LOCAL_WEIGHT_DIR = "/path/to/dinov3-weights"  # folder containing model.safetensors
```

Model weights, datasets, checkpoints, and generated logs are not committed to
this repository.

## Dataset Structure

```text
<dataroot>/
  train/
    <class_name>/        # e.g. car, cat, sdv1.4
      0_real/
        *.jpg/png
      1_fake/
        *.jpg/png
  test/
    <generator_name>/    # e.g. DALLE-3, FLUX1-dev, StyleGAN3
      0_real/
        *.jpg/png
      1_fake/
        *.jpg/png
```

## Train

```bash
python run_train.py --dataroot /path/to/data --nproc_per_node 3
```

By default, training loads images from the folders under `--dataroot`. To use a
deterministic training order, pass a sample list explicitly:

```bash
python run_train.py --dataroot /path/to/data --sample_list ./data/train_samples.example.txt
```

Sample lists use tab-separated `path<TAB>label` rows. Relative paths are resolved
from `--dataroot`; absolute local paths should not be committed.

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--sample_list` | `None` | Optional sample list for deterministic ordering |
| `--niter` | `1` | Training epochs |
| `--batch_size` | `16` | Batch size per GPU |
| `--lr` | `5e-5` | Learning rate |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA scaling factor |
| `--lora_dropout` | `0.3` | LoRA dropout |
| `--ase_threshold` | `0.75` | Attention erasure threshold |
| `--npe_epsilon` | `0.0005` | Adversarial perturbation strength |
| `--clean_loss_weight` | `0.1` | Clean CE loss weight |
| `--ase_loss_weight` | `0.45` | ASE loss weight |
| `--npe_loss_weight` | `0.45` | NPE loss weight |
| `--weight_decay` | `0.001` | AdamW weight decay |

Checkpoints are saved to `./checkpoints/<experiment_name>/`.

## Test

```bash
python test.py --test_dataroot /path/to/data/test --model_path /path/to/checkpoint.pth --gpu_id 0
```

## Download Weights

You can download the model weights from Google Drive:
- [Model Weights](https://drive.google.com/file/d/1ItUkYXH6d9L3MZmCUT6uxHwP5wePtHlC/view?usp=sharing)

## Third-Party Code

This project includes DINOv3-derived code under `models/dinov3`. Keep the
corresponding DINOv3 license terms with any public release or redistribution.
