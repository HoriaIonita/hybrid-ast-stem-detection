# Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection

This repository contains the official PyTorch implementation for the paper: **"Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection with DSP Stabilization."** 

It provides the complete pipeline for dataset preprocessing, model training (Proposed AST, Enhanced ResNet-18, Enhanced Custom CNN), DSP-stabilized inference, and the custom "Honest Scoring System" for robust multi-seed evaluation on the MUSDB18 dataset.

## 🚀 Quickstart & Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/HoriaIonita/hybrid-ast-stem-detection.git](https://github.com/YOUR-USERNAME/hybrid-ast-stem-detection.git)
cd hybrid-ast-stem-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📂 Repository Structure

* `train_ast.py` - Core training script for the Proposed AST.
* `evaluate_dsp.py` - Post-processing inference script with the w=5 non-linear median filter.
* `cnn_enhanced_dsp.py` - Training and evaluation for the Enhanced Custom CNN competitor.
* `resnet18_enhanced_dsp.py` - Training and evaluation for the Enhanced Pre-Trained ResNet-18 competitor.
* `preprocess_musdb.py` - High-speed I/O optimization pipeline for the dataset.
* `run_ablations.py` - Automated multi-seed execution of the ablation study.
* `generate_ablation_tables.py` - Aggregates raw CSV outputs into manuscript-ready statistical tables.
* `honest_scoring_system.py` - Generates the Honest Ranking Bar/Radar charts with instability penalties.

## 💽 Dataset Preprocessing
This framework relies on the [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html). To bypass CPU/Disk I/O bottlenecks during high-speed training, the `.stem.mp4` files must be preprocessed into mono, 16kHz memory-mapped `.npy` arrays.

```bash
# Point the script to your downloaded MUSDB18 folder
python preprocess_musdb.py --source_dir ./musdb18 --target_dir ./musdb18_npy
```

## 🧠 Training the Models
All models are evaluated across 3 independent random seeds (42, 123, 777) to ensure statistical rigor and to prevent initialization bias.

**To train the Proposed AST framework:**
```bash
python train_ast.py
```
*Note: The script automatically handles Dynamic Zero-Padding, Stochastic Source Dropping (p=0.3), and Asymmetric Loss (gamma=1.0) optimization.*

**To train the Enhanced Competitor Baselines:**
```bash
python cnn_enhanced_dsp.py
python resnet18_enhanced_dsp.py
```

## ⚡ Inference & DSP Post-Processing
To run inference on the test set using the calibrated thresholds and the w=5 non-linear DSP median filter:
```bash
python evaluate_dsp.py
```

## 📊 The "Honest Scoring System" & Ablation Studies
To reproduce the exact ablation studies and generate the radar/bar charts utilized in the manuscript:

1. **Run the automated ablation suite:**
```bash
python run_ablations.py
```

2. **Generate the formatted statistical tables:**
```bash
python generate_ablation_tables.py
```

3. **Generate the Honest Ranking graphs:**
```bash
python honest_scoring_system.py
```

## 📝 Citation
If you utilize this code or the Honest Scoring metric in your research, please cite our paper:
```bibtex
@article{ionita2026ast,
  title={Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection with DSP Stabilization},
  author={Ioniță, Horia-Sebastian and Popa, Victor and Paleologu, Constantin},
  journal={TBD},
  year={2026}
}
```