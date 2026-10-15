# Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection

Official repository for the MDPI manuscript: **"Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection with DSP Stabilization"**.

This repository contains the complete, reproducible training and evaluation pipeline for the proposed AST framework and its associated baselines.

## Quickstart & Installation

1. **Clone the repository:**
```bash
git clone https://github.com/HoriaIonita/hybrid-ast-stem-detection.git
cd hybrid-ast-stem-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Repository Structure & File Overview

### 1. Final Proposed Model & Pre-Trained Baselines (Tables 1 & 2)
* `proposed_ast_revised.py`: The finalized AST framework featuring causal DSP ($w=5$), PR-curve calibration, and track-level bootstrapping (95% CIs).
* `panns_enhanced_dsp.py`: The official AudioSet-pretrained PANNs CNN14 baseline.
* `cnn_enhanced_dsp.py`: Custom 4-layer CNN baseline trained from scratch (evaluated with centered DSP to provide a theoretical look-ahead advantage).
* `resnet18_enhanced_dsp.py`: ImageNet-pretrained ResNet-18 baseline (evaluated with centered DSP).
* `evaluate_flops_and_sensitivity.py`: Calculates parameter/FLOPs complexity and generates the ground-truth energy threshold sensitivity sweep.

### 2. Ablation Study Scripts (Table 3)
*To ensure full reproducibility of the comprehensive ablation ledger (Table 3), the original development and intermediate scripts have been preserved:*
* `proposed_ast.py`: The base AST architecture evaluated without the DSP stabilization stage (Raw Probabilities).
* `proposed_ast_dsp.py`: The AST architecture evaluated utilizing the centered (non-causal) median filter.
* `run_ablations.py`: Automated multi-seed execution of the ablation study.
* `generate_ablation_tables.py`: Aggregates raw CSV outputs into manuscript-ready statistical tables.

### 3. Data Preprocessing
* `preprocess_musdb.py`: High-speed I/O optimization pipeline for the MUSDB18 dataset.

## Dataset Preprocessing
This framework relies on the [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html). To bypass CPU/Disk I/O bottlenecks during high-speed training, the `.stem.mp4` files must be preprocessed into mono, 16kHz memory-mapped `.npy` arrays.

```bash
# Point the script to your downloaded MUSDB18 folder
python preprocess_musdb.py --source_dir ./musdb18 --target_dir ./musdb18_npy
```

## Training the Models
All models are evaluated across 3 independent random seeds (42, 123, 777) to ensure statistical rigor and to prevent initialization bias.

**To train the Proposed AST framework:**
```bash
python proposed_ast_revised.py
```
*Note: The script automatically handles Dynamic Zero-Padding, Stochastic Source Dropping (p=0.3), Asymmetric Loss (gamma=1.0) optimization, Causal DSP, and Track-Level Bootstrapping.*

**To train the Pre-Trained PANNs Baseline:**
```bash
python panns_enhanced_dsp.py
```

**To train the Competitor Baselines:**
```bash
python cnn_enhanced_dsp.py
python resnet18_enhanced_dsp.py
```

## Evaluation & Profiling
To evaluate computational complexity (FLOPs & Parameters) and generate the ground-truth energy threshold sensitivity sweep:
```bash
python evaluate_flops_and_sensitivity.py
```

## Citation
If you utilize this code in your research, please cite our paper:
```bibtex
@article{ionita2026ast,
  title={Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection with DSP Stabilization},
  author={Ioniță, Horia-Sebastian and Popa, Victor and Paleologu, Constantin},
  journal={Electronics},
  year={2026}
}
```
