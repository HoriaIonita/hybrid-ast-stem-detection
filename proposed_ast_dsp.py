# -*- coding: utf-8 -*-
"""
Title: Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection
Authors: Horia-Sebastian Ioniță

Description:
DSP Post-Processing Script for the trained Hybrid AST framework.
- Loads the optimized .pt weights directly from disk.
- Bypasses training and runs pure inference.
- Applies a non-linear Median Filter (w=5) over raw binary predictions.
- Calculates and saves the stabilized final metrics.
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, f1_score, 
                             multilabel_confusion_matrix, balanced_accuracy_score, 
                             matthews_corrcoef, average_precision_score)
import torchaudio.transforms as T
import torch.nn.functional as F
from transformers import ASTModel
from scipy.signal import medfilt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# --- CONFIGURATION ---
seeds = [42, 123, 777]
models_dir = "Results_Proposed_AST"
results_dir = "Results_DSP"
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "proposed_ast_dsp_metrics.csv")

max_len = 80000
all_instruments = ['drums', 'bass', 'other', 'vocals']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def console_log(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {text}")
    sys.stdout.flush()

# --- DSP MODULE ---
def dsp_post_processing(probs_matrix, thresholds, median_window=5):
    """
    Applies Median Filtering to eliminate transient noise (spikes)
    and bridge micro-silences without altering logical decision thresholds.
    """
    num_segments, num_instruments = probs_matrix.shape
    cleaned_predictions = np.zeros_like(probs_matrix)
    
    for i in range(num_instruments):
        raw_probs = probs_matrix[:, i]
        binary_raw = (raw_probs > thresholds[i]).astype(float)
        cleaned_predictions[:, i] = medfilt(binary_raw, kernel_size=median_window)
            
    return cleaned_predictions

# --- DATASET & TRANSFORMS (Inference Only) ---
class LogMelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=160, n_frames=501):
        self.n_frames = n_frames
        self.transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        mel = self.amplitude_to_db(self.transform(waveform))
        return F.pad(mel, (0, max(0, self.n_frames - mel.shape[-1])))[:, :, :self.n_frames]

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, sample_rate=16000, max_len=80000):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.max_len = max_len
        self.instruments = ('drums','bass','other','vocals')
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

        start_idx = (num_samples - self.max_len) // 2 if num_samples > self.max_len else 0
        stems_slice = np.array(stems[:, start_idx : start_idx + self.max_len]) if num_samples > self.max_len else np.pad(np.array(stems), ((0,0), (0, self.max_len - num_samples)))

        waveform = torch.tensor(stems_slice[0]).float().unsqueeze(0)

        if self.transform:
            waveform = self.transform(waveform)
            if torch.std(waveform) > 0: waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
            if waveform.dim() == 2: waveform = waveform.unsqueeze(0)

        return waveform, self.instrument_presence_mask(stems_slice)

# --- ARCHITECTURE ---
class TransferAST(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        if x.shape[1] < 1024: x = F.pad(x, (0, 0, 0, 1024 - x.shape[1]))
        outputs = self.ast(x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# --- EVALUATION ---
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
    return {"thresholds": np.array(thresholds)}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    console_log(f"\n{'='*70}\n[START] INFERENCE WITH DSP POST-PROCESSING\n{'='*70}")
    transform = LogMelSpectrogramTransform()
    
    train_dataset = MUSDB18Dataset("musdb18_npy", split='train', transform=transform)
    _, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
    val_loader_clean = DataLoader(Subset(train_dataset, val_idx), batch_size=10, shuffle=False)
    
    test_dataset = MUSDB18Dataset("musdb18_npy", split='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = TransferAST(num_classes=4).to(device)
    
    for seed in seeds:
        exp_name = "Proposed_AST_DSP"
        model_filename = f"model_Proposed_AST_Seed{seed}.pt"
        model_path = os.path.join(models_dir, model_filename)
        
        if not os.path.exists(model_path):
            console_log(f"[ERROR] Could not find file {model_path}. Check the exact naming.")
            continue
            
        console_log(f"\n--- Loading model for Seed {seed} ---")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        console_log("Calibrating thresholds...")
        thresholds = evaluate_val(model, val_loader_clean, device, all_instruments)["thresholds"]
        console_log(f"Optimal thresholds found: {thresholds}")
        
        console_log("Running on test set...")
        y_true, probs_agg = [], []
        hop_length = max_len // 2 
        
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
        
        # --- APPLY DSP ---
        console_log("Applying DSP logic (Median Filter)...")
        y_pred = dsp_post_processing(probs_agg, thresholds, median_window=5)
        
        df_metrics = detailed_metrics(y_true, y_pred, probs_agg, all_instruments)
        console_log(f"FINAL METRICS DSP (Seed {seed})")
        console_log("\n" + df_metrics.round(3).to_string(index=False))
        
        save_to_master_csv(df_metrics, exp_name, seed, ms_per_seg)
        
    console_log("\n[SUCCESS] DSP evaluation finished on all available seeds!")