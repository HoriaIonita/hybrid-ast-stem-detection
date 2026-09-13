# -*- coding: utf-8 -*-
"""
Title: Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection
Authors: Horia-Sebastian Ioniță

Description:
Training and evaluation script for the Enhanced Custom CNN.
Employs the exact same pipeline advantages as the SOTA framework
to guarantee a 100% fair mathematical comparison:
- Loss: Asymmetric Loss (ASL) with gamma_neg = 1.0
- Stochastic Regularization: Source Dropping = 0.3
- Augmentations: SpecAugment + Physical Waveform
- DSP: Median Filter W=5
"""

import os
import sys
import time
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, classification_report, f1_score, 
                             multilabel_confusion_matrix, balanced_accuracy_score, 
                             matthews_corrcoef, average_precision_score)
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from scipy.signal import medfilt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# --- CNN CONFIGURATION ---
seeds = [42, 123, 777]
results_dir = "Results_CNN_Enhanced"
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "cnn_enhanced_metrics.csv")

OPTIMAL_GAMMA_NEG = 1.0
OPTIMAL_DROP_PROB = 0.3

epochs = 25
batch_size = 10
early_stopping_patience = 6
clip_grad_norm = 1.0
warmup_epochs = 3
max_len = 80000
all_instruments = ['drums', 'bass', 'other', 'vocals']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def console_log(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {text}")
    sys.stdout.flush()
    with open(os.path.join(results_dir, "cnn_execution.log"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

# --- AUGMENTATIONS ---
def augment(waveform, sample_rate):
    if random.random() < 0.5:
        rate = random.uniform(0.85, 1.15)
        new_length = int(waveform.shape[1] / rate)
        waveform = F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
    if random.random() < 0.5:
        n_steps = random.choice([-2, -1, 1, 2])
        new_sr = int(sample_rate * (2 ** (n_steps / 12)))
        waveform = torchaudio.functional.resample(waveform, sample_rate, new_sr)
        waveform = torchaudio.functional.resample(waveform, new_sr, sample_rate)
    if random.random() < 0.3:
        waveform += 0.005 * random.random() * torch.randn_like(waveform)
    if random.random() < 0.3:
        gain_db = random.uniform(-6, 6)
        waveform *= (10 ** (gain_db/20))
    return waveform

def spec_augment(mel_spec, freq_mask_param=20, time_mask_param=30):
    mel_spec = mel_spec.clone()
    num_mels, num_frames = mel_spec.shape[-2], mel_spec.shape[-1]
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, num_mels - f)
    mel_spec[:, f0:f0+f, :] = 0
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, num_frames - t)
    mel_spec[:, :, t0:t0+t] = 0
    return mel_spec

# --- DSP MODULE ---
def dsp_post_processing(probs_matrix, thresholds, median_window=5):
    num_segments, num_instruments = probs_matrix.shape
    cleaned_predictions = np.zeros_like(probs_matrix)
    for i in range(num_instruments):
        raw_probs = probs_matrix[:, i]
        binary_raw = (raw_probs > thresholds[i]).astype(float)
        cleaned_predictions[:, i] = medfilt(binary_raw, kernel_size=median_window)
    return cleaned_predictions

# --- DATASET & TRANSFORMS ---
class LogMelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=160, n_frames=501):
        self.n_frames = n_frames
        self.transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        mel = self.amplitude_to_db(self.transform(waveform))
        return F.pad(mel, (0, max(0, self.n_frames - mel.shape[-1])))[:, :, :self.n_frames]

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, sample_rate=16000, 
                 max_len=80000, instruments=('drums','bass','other','vocals'), negative_prob=0.0, augment_flag=True):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.instruments = instruments
        self.negative_prob = negative_prob
        self.augment_flag = augment_flag
        self.audio_paths = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.npy')])

    def __len__(self): return len(self.audio_paths)

    def instrument_presence_mask(self, stems_slice, threshold_ratio=0.03):
        mixture_energy = np.mean(stems_slice[0] ** 2)
        dynamic_threshold = mixture_energy * threshold_ratio + 1e-9
        index_map = {'drums':1, 'bass':2, 'other':3, 'vocals':4}
        presence_flags = []
        for inst in self.instruments:
            idx = index_map[inst]
            presence_flags.append(1.0 if np.mean(stems_slice[idx] ** 2) > dynamic_threshold else 0.0)
        return torch.tensor(presence_flags, dtype=torch.float32)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        stems = np.load(path, mmap_mode='r')
        num_samples = stems.shape[1]

        if self.split == 'test': return torch.tensor(np.array(stems)).float()

        if num_samples > self.max_len:
            start_idx = np.random.randint(0, num_samples - self.max_len) if self.split == 'train' else (num_samples - self.max_len) // 2
            stems_slice = np.array(stems[:, start_idx : start_idx + self.max_len])
        else:
            stems_slice = np.array(stems)
            stems_slice = np.pad(stems_slice, ((0,0), (0, self.max_len - num_samples)))

        # SOURCE DROPPING 
        if self.augment_flag and self.negative_prob > 0:
            if random.random() < 0.05: stems_slice[1:] = 0.0
            else:
                for i in range(1, 5): 
                    if random.random() < self.negative_prob: stems_slice[i] = 0.0
            stems_slice[0] = np.sum(stems_slice[1:], axis=0)

        waveform = torch.tensor(stems_slice[0]).float().unsqueeze(0)

        if self.augment_flag and torch.sum(torch.abs(waveform)) > 0:
            waveform = augment(waveform, self.sample_rate)

        if self.transform:
            waveform = self.transform(waveform)
            if torch.std(waveform) > 0: waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
            if waveform.dim() == 2: waveform = waveform.unsqueeze(0)
            if self.augment_flag: waveform = spec_augment(waveform)

        return waveform, self.instrument_presence_mask(stems_slice)

# --- ARCHITECTURE ---
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(EnhancedCNN, self).__init__()
        
        # Input Mel Spectrogram: [Batch, 1, 128, 501]
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Feature output [Batch, 256, H', W']
        x = self.features(x)
        # GAP reduces spatially to [Batch, 256, 1, 1]
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- LOSS (Asymmetric 1.0) & UTILS ---
def asymmetric_loss(output, target, gamma_pos=0.0, gamma_neg=1.0):
    x_sigmoid = torch.sigmoid(output)
    loss_pos = target * torch.log(x_sigmoid + 1e-8) * (1 - x_sigmoid) ** gamma_pos
    loss_neg = (1 - target) * torch.log(1 - x_sigmoid + 1e-8) * x_sigmoid ** gamma_neg
    return -torch.mean(loss_pos + loss_neg)

def detailed_metrics(y_true, y_pred, probs, all_instruments):
    cm_per_class = multilabel_confusion_matrix(y_true, y_pred)
    rows = []
    for i, inst in enumerate(all_instruments):
        tn, fp, fn, tp = cm_per_class[i].ravel()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        bal_acc = balanced_accuracy_score(y_true[:, i], y_pred[:, i])
        mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        auprc_neg = average_precision_score(1 - y_true[:, i], 1 - probs[:, i])

        rows.append({"Instrument": inst, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "Precision": precision, "Recall": recall, "F1": f1,
                     "Specificity": specificity, "FPR": fpr, "BalAcc": bal_acc, 
                     "MCC": mcc, "Neg_AUPRC": auprc_neg})
    return pd.DataFrame(rows)

def save_to_master_csv(df_metrics, exp_name, seed, inference_ms):
    df_metrics.insert(0, "Seed", seed)
    df_metrics.insert(1, "Experiment", exp_name)
    df_metrics["InferenceMsPerSeg"] = inference_ms
    file_exists = os.path.isfile(csv_path)
    df_metrics.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def evaluate_val(model, dataloader, device, all_instruments):
    model.eval()
    y_true, probs = [], []
    with torch.no_grad():
        for mix, label in dataloader:
            out = model(mix.to(device))
            y_true.append(label.numpy())
            probs.append(torch.sigmoid(out).cpu().numpy())
            
    y_true, probs = np.vstack(y_true), np.vstack(probs)
    thresholds = []
    for i in range(len(all_instruments)):
        p, r, t = precision_recall_curve(y_true[:, i], probs[:, i])
        f1s = 2 * (p * r) / (p + r + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        thresholds.append(t[best_idx] if len(t) > best_idx else 0.5)
        
    y_pred = (probs > np.array(thresholds)).astype(float)
    return {"macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0), "thresholds": np.array(thresholds)}

def calibrate_and_evaluate(model, val_loader_clean, test_loader, test_dataset, device, transform, seed, exp_name):
    console_log(f"[{exp_name}] Calibrating Mathematical Thresholds on Clean Validation Set...")
    val_metrics = evaluate_val(model, val_loader_clean, device, all_instruments)
    thresholds = val_metrics["thresholds"]
    console_log(f"[{exp_name}] Optimal Thresholds: {thresholds}")
    
    console_log(f"[{exp_name}] Running Segment-Level Final Test Evaluation (Overlap: True)...")
    model.eval()
    y_true, probs_agg = [], []
    hop_length = max_len // 2  # Overlap 50%
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for full_stems in test_loader: 
            total_samples = full_stems.shape[-1]
            for start in range(0, total_samples - max_len + 1, hop_length):
                stem_segment = full_stems[0, :, start:start+max_len].numpy()
                label = test_dataset.instrument_presence_mask(stem_segment, threshold_ratio=0.03)
                
                waveform = torch.tensor(stem_segment[0]).float().unsqueeze(0)
                mel = transform(waveform)
                if torch.std(mel) > 0: mel = (mel - mel.mean()) / (mel.std() + 1e-9)
                
                out = model(mel.unsqueeze(0).to(device))
                probs_agg.append(torch.sigmoid(out).cpu().numpy()[0])
                y_true.append(label.numpy())
            
    ms_per_seg = ((time.perf_counter() - start_time) / len(probs_agg)) * 1000
    y_true, probs_agg = np.vstack(y_true), np.vstack(probs_agg)
    
    console_log(f"[{exp_name}] Applying DSP logic...")
    y_pred = dsp_post_processing(probs_agg, thresholds, median_window=5)

    df_metrics = detailed_metrics(y_true, y_pred, probs_agg, all_instruments)
    console_log(f"[{exp_name}] FINAL METRICS (Seed {seed}) | Infer: {ms_per_seg:.2f}ms/seg")
    console_log("\n" + df_metrics.round(3).to_string(index=False))
    
    save_to_master_csv(df_metrics, exp_name, seed, ms_per_seg)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    console_log(f"\n{'='*70}\n[START] INITIALIZING RIGOROUS ENHANCED CNN (LSE=1.0, Drop=0.3, DSP)\n{'='*70}")
    transform = LogMelSpectrogramTransform()
    
    for seed in seeds:
        exp_name = "CNN_Enhanced_DSP"
        run_key = f"{exp_name}_Seed{seed}"
        
        console_log(f"\n--- Running for Seed: {seed} ---")
        
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        train_dataset = MUSDB18Dataset("musdb18_npy", split='train', transform=transform, negative_prob=OPTIMAL_DROP_PROB, augment_flag=True)
        train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
        
        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_dataset_clean = MUSDB18Dataset("musdb18_npy", split='train', transform=transform, augment_flag=False)
        val_loader_clean = DataLoader(Subset(val_dataset_clean, val_idx), batch_size=batch_size, shuffle=False)
        test_dataset = MUSDB18Dataset("musdb18_npy", split='test', transform=None, augment_flag=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        model = EnhancedCNN(num_classes=4).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        save_path = os.path.join(results_dir, f"model_{run_key}.pt")
        best_val_macro_f1 = 0; patience_counter = 0
        
        # TRAINING LOOP
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            if epoch < warmup_epochs:
                for pg in optimizer.param_groups: pg['lr'] = 1e-4 * min(1.0, float(epoch + 1) / warmup_epochs)
                    
            for step, (mix, label) in enumerate(train_loader):
                mix, label = mix.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(mix)
                loss = asymmetric_loss(output, label, gamma_neg=OPTIMAL_GAMMA_NEG)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step(); running_loss += loss.item()
            
            if epoch >= warmup_epochs: scheduler.step()
                
            val_macro_f1 = evaluate_val(model, val_loader_clean, device, all_instruments)["macro_f1"]
            console_log(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val F1: {val_macro_f1:.4f}")
            
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1; patience_counter = 0; torch.save(model.state_dict(), save_path)
            else: patience_counter += 1
                
            if patience_counter >= early_stopping_patience: 
                console_log(f"   [EARLY STOPPING] Triggered.")
                break
        
        # FINAL EVALUATION
        model.load_state_dict(torch.load(save_path, weights_only=True))
        calibrate_and_evaluate(model, val_loader_clean, test_loader, test_dataset, device, transform, seed, exp_name)
        
    console_log("\n[SUCCESS] Execution for Enhanced CNN (3 seeds) fully completed!")