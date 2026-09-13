# -*- coding: utf-8 -*-
"""
Title: Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection
Authors: Horia-Sebastian Ioniță

Description:
Experimental automation system across 3 seeds with persistent JSON checkpointing.
Systematically tests all ablation configurations: NoPretrain, NoOverlap, BCE Loss, etc.
"""

import os
import sys
import time
import json
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
from transformers import ASTModel, ASTConfig
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# --- MASTER CONFIGURATION ---
seeds = [42, 123, 777]
results_master_dir = "Results_Ablation_Master"
os.makedirs(results_master_dir, exist_ok=True)

checkpoint_json_path = os.path.join(results_master_dir, "pipeline_state.json")
master_csv_path = os.path.join(results_master_dir, "master_ablation_metrics.csv")

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
    with open(os.path.join(results_master_dir, "master_execution.log"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

# --- CHECKPOINT STATE MANAGEMENT ---
def load_pipeline_state():
    if os.path.exists(checkpoint_json_path):
        with open(checkpoint_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_runs": [], "last_updated": ""}

def save_pipeline_state(state, completed_run_key):
    state["completed_runs"].append(completed_run_key)
    state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(checkpoint_json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

# --- AUGMENTATION ---
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
class TransferAST(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        if pretrained:
            self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        else:
            # Blind Architecture (Ablation_NoPretrain)
            config = ASTConfig()
            self.ast = ASTModel(config)
            
        self.classifier = nn.Linear(768, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        if x.shape[1] < 1024: x = F.pad(x, (0, 0, 0, 1024 - x.shape[1]))
        outputs = self.ast(x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# --- LOSS & UTILS ---
def asymmetric_loss(output, target, gamma_pos=0.0, gamma_neg=2.0):
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
    
    file_exists = os.path.isfile(master_csv_path)
    df_metrics.to_csv(master_csv_path, mode='a', header=not file_exists, index=False)

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

def calibrate_and_evaluate(model, val_loader_clean, test_loader, test_dataset, device, transform, seed, use_overlap, exp_name):
    console_log(f"[{exp_name}] Calibrating Mathematical Thresholds on Clean Validation Set...")
    val_metrics = evaluate_val(model, val_loader_clean, device, all_instruments)
    thresholds = val_metrics["thresholds"]
    console_log(f"[{exp_name}] Optimal Thresholds: {thresholds}")
    
    console_log(f"[{exp_name}] Running Segment-Level Final Test Evaluation (Overlap: {use_overlap})...")
    model.eval()
    y_true, probs_agg = [], []
    
    # Ablation NoOverlap
    hop_length = (max_len // 2) if use_overlap else max_len 
    
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
    y_pred = (probs_agg > thresholds).astype(float)

    df_metrics = detailed_metrics(y_true, y_pred, probs_agg, all_instruments)
    console_log(f"[{exp_name}] FINAL METRICS (Seed {seed}) | Infer: {ms_per_seg:.2f}ms/seg")
    console_log("\n" + df_metrics.round(3).to_string(index=False))
    
    save_to_master_csv(df_metrics, exp_name, seed, ms_per_seg)

# --- EXPERIMENTAL CONFIGURATION MATRIX ---
experiments = [
    {"name": "Proposed_AST", "loss": "LSE", "gamma_neg": 2.0, "drop": 0.5, "aug": True, "overlap": True, "pretrained": True},
    {"name": "Sensitivity_Loss_LSE3.0", "loss": "LSE", "gamma_neg": 3.0, "drop": 0.5, "aug": True, "overlap": True, "pretrained": True},
    {"name": "Sensitivity_Loss_LSE1.0", "loss": "LSE", "gamma_neg": 1.0, "drop": 0.5, "aug": True, "overlap": True, "pretrained": True},
    {"name": "Sensitivity_Loss_BCE", "loss": "BCE", "gamma_neg": None, "drop": 0.5, "aug": True, "overlap": True, "pretrained": True},
    {"name": "Sensitivity_Drop_0.3", "loss": "LSE", "gamma_neg": 2.0, "drop": 0.3, "aug": True, "overlap": True, "pretrained": True},
    {"name": "Sensitivity_Drop_0.0", "loss": "LSE", "gamma_neg": 2.0, "drop": 0.0, "aug": True, "overlap": True, "pretrained": True},
    {"name": "Ablation_NoAugment", "loss": "LSE", "gamma_neg": 2.0, "drop": 0.5, "aug": False, "overlap": True, "pretrained": True},
    {"name": "Ablation_NoPretrain", "loss": "LSE", "gamma_neg": 2.0, "drop": 0.5, "aug": True, "overlap": True, "pretrained": False},
    {"name": "Ablation_NoOverlap", "loss": "LSE", "gamma_neg": 2.0, "drop": 0.5, "aug": True, "overlap": False, "pretrained": True}
]

if __name__ == "__main__":
    pipeline_state = load_pipeline_state()
    console_log(f"Checkpoint system initialized. Previously completed runs: {len(pipeline_state['completed_runs'])}")
    transform = LogMelSpectrogramTransform()
    
    for seed in seeds:
        for exp in experiments:
            run_key = f"{exp['name']}_Seed{seed}"
            
            if run_key in pipeline_state["completed_runs"]:
                console_log(f"[SKIP] Experiment {run_key} was already completed.")
                continue
                
            console_log(f"\n{'='*60}\n[START] Initializing experiment: {run_key}\n{'='*60}")
            
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Dataloaders
            train_dataset = MUSDB18Dataset("musdb18_npy", split='train', transform=transform, negative_prob=exp['drop'], augment_flag=exp['aug'])
            train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
            train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
            
            val_dataset_clean = MUSDB18Dataset("musdb18_npy", split='train', transform=transform, augment_flag=False)
            val_loader_clean = DataLoader(Subset(val_dataset_clean, val_idx), batch_size=batch_size, shuffle=False)
            
            test_dataset = MUSDB18Dataset("musdb18_npy", split='test', transform=None, augment_flag=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            model = TransferAST(num_classes=4, pretrained=exp['pretrained']).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
            
            save_path = os.path.join(results_master_dir, f"model_{run_key}.pt")
            best_val_macro_f1 = 0; patience_counter = 0
            
            # Logic for NoOverlap (reusing SOTA model for speed)
            if exp['name'] == "Ablation_NoOverlap":
                sota_model_path = os.path.join(results_master_dir, f"model_SOTA_Seed{seed}.pt")
                if os.path.exists(sota_model_path):
                    console_log(f"Reloading SOTA model ({seed}) for NoOverlap evaluation.")
                    save_path = sota_model_path
                else:
                    console_log(f"[WARN] SOTA model missing for seed {seed}. Running full training out of necessity.")
            
            # TRAINING LOOP
            if "NoOverlap" not in exp['name'] or not os.path.exists(sota_model_path):
                for epoch in range(epochs):
                    model.train()
                    running_loss = 0.0
                    if epoch < warmup_epochs:
                        for pg in optimizer.param_groups: pg['lr'] = 1e-4 * min(1.0, float(epoch + 1) / warmup_epochs)
                            
                    for step, (mix, label) in enumerate(train_loader):
                        mix, label = mix.to(device), label.to(device)
                        optimizer.zero_grad()
                        output = model(mix)
                        
                        if exp['loss'] == "BCE": loss = nn.BCEWithLogitsLoss()(output, label)
                        else: loss = asymmetric_loss(output, label, gamma_neg=exp['gamma_neg'])
                            
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        optimizer.step(); running_loss += loss.item()
                    
                    if epoch >= warmup_epochs: scheduler.step()
                        
                    val_macro_f1 = evaluate_val(model, val_loader_clean, device, all_instruments)["macro_f1"]
                    console_log(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val F1: {val_macro_f1:.4f}")
                    
                    if val_macro_f1 > best_val_macro_f1:
                        best_val_macro_f1 = val_macro_f1; patience_counter = 0; torch.save(model.state_dict(), save_path)
                    else: patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience: break
            
            # FINAL EVALUATION
            model.load_state_dict(torch.load(save_path, weights_only=True))
            calibrate_and_evaluate(model, val_loader_clean, test_loader, test_dataset, device, transform, seed, exp['overlap'], run_key)
            
            save_pipeline_state(pipeline_state, run_key)
            
    console_log("ALL 27 EXPERIMENTAL RUNS COMPLETED SUCCESSFULLY.")