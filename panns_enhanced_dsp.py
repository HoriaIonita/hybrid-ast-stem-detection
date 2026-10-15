# -*- coding: utf-8 -*-
"""
Title: Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection
Authors: Horia-Sebastian Ioniță

Description: PANNs (CNN14) Audio-Pretrained Baseline for MDPI Review
"""

import os
import sys
import time
import torch
import urllib.request
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, f1_score, 
                             multilabel_confusion_matrix, balanced_accuracy_score, 
                             matthews_corrcoef, average_precision_score,
                             PrecisionRecallDisplay, ConfusionMatrixDisplay)
from sklearn.calibration import CalibrationDisplay
import torchaudio.transforms as T
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# --- CONFIGURATION ---
seeds = [42, 123, 777]
results_dir = "Results_PANNs_Baseline"
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, "panns_metrics_revision.csv")

epochs = 25
batch_size = 10
early_stopping_patience = 6
clip_grad_norm = 1.0
warmup_epochs = 3
max_len = 80000
all_instruments = ['drums', 'bass', 'other', 'vocals']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def console_log(text):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {text}")
    sys.stdout.flush()
    with open(os.path.join(results_dir, "panns_execution.log"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

# --- CAUSAL DSP & EVALUATION ---
def causal_dsp_post_processing(probs_matrix, thresholds, median_window=5):
    num_segments, num_instruments = probs_matrix.shape
    cleaned_predictions = np.zeros_like(probs_matrix)
    for i in range(num_instruments):
        raw_probs = probs_matrix[:, i]
        binary_raw = (raw_probs > thresholds[i]).astype(float)
        padded_binary = np.pad(binary_raw, (median_window - 1, 0), mode='constant')
        for t in range(num_segments):
            window_slice = padded_binary[t : t + median_window]
            cleaned_predictions[t, i] = np.median(window_slice)
    return cleaned_predictions

def calculate_switching_and_errors(y_true, y_pred, hop_length_sec=2.5):
    switching_rate = np.mean(np.abs(np.diff(y_pred)))
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    true_onsets = np.where(true_diff == 1)[0]
    pred_onsets = np.where(pred_diff == 1)[0]
    true_offsets = np.where(true_diff == -1)[0]
    pred_offsets = np.where(pred_diff == -1)[0]
    
    onset_errors, offset_errors = [], []
    for t_onset in true_onsets:
        if len(pred_onsets) > 0:
            nearest_pred = pred_onsets[np.argmin(np.abs(pred_onsets - t_onset))]
            onset_errors.append(np.abs(t_onset - nearest_pred) * hop_length_sec)
    for t_offset in true_offsets:
        if len(pred_offsets) > 0:
            nearest_pred = pred_offsets[np.argmin(np.abs(pred_offsets - t_offset))]
            offset_errors.append(np.abs(t_offset - nearest_pred) * hop_length_sec)
            
    return switching_rate, np.mean(onset_errors) if onset_errors else 0.0, np.mean(offset_errors) if offset_errors else 0.0

def track_level_bootstrap_metrics(y_true, y_pred_raw, y_pred_dsp, probs, track_ids, exp_name, seed, hop_sec=2.5, n_bootstraps=1000):
    unique_tracks = np.unique(track_ids)
    rows = []
    
    for i, inst in enumerate(all_instruments):
        inst_name = inst.capitalize()
        inst_y_true = y_true[:, i]
        inst_y_pred = y_pred_dsp[:, i] 
        inst_y_pred_raw = y_pred_raw[:, i] 
        inst_probs = probs[:, i]
        
        tn, fp, fn, tp = multilabel_confusion_matrix(y_true, y_pred_dsp)[i].ravel()
        class_prevalence = (tp + fn) / len(inst_y_true)
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        base_f1 = f1_score(inst_y_true, inst_y_pred, zero_division=0)
        specificity = tn / (tn + fp + 1e-9)
        base_mcc = matthews_corrcoef(inst_y_true, inst_y_pred)
        bal_acc = balanced_accuracy_score(inst_y_true, inst_y_pred)
        base_auprc = average_precision_score(inst_y_true, inst_probs)
        base_neg_auprc = average_precision_score(1 - inst_y_true, 1 - inst_probs)
        
        raw_switching, _, _ = calculate_switching_and_errors(inst_y_true, inst_y_pred_raw, hop_sec)
        dsp_switching, onset_err, offset_err = calculate_switching_and_errors(inst_y_true, inst_y_pred, hop_sec)
        
        boot_f1s, boot_mccs = [], []
        for _ in range(n_bootstraps):
            sample_tracks = np.random.choice(unique_tracks, size=len(unique_tracks), replace=True)
            sample_indices = np.concatenate([np.where(track_ids == t)[0] for t in sample_tracks])
            b_true = inst_y_true[sample_indices]
            b_pred = inst_y_pred[sample_indices]
            if len(np.unique(b_true)) > 1: 
                boot_f1s.append(f1_score(b_true, b_pred, zero_division=0))
                boot_mccs.append(matthews_corrcoef(b_true, b_pred))
        
        f1_ci = (np.percentile(boot_f1s, 2.5), np.percentile(boot_f1s, 97.5)) if boot_f1s else (base_f1, base_f1)
        mcc_ci = (np.percentile(boot_mccs, 2.5), np.percentile(boot_mccs, 97.5)) if boot_mccs else (base_mcc, base_mcc)

        rows.append({
            "Instrument": inst_name,
            "Prevalence": class_prevalence,
            "Raw_Switch_Rate": raw_switching, "DSP_Switch_Rate": dsp_switching,
            "Onset_Err_Sec": onset_err, "Offset_Err_Sec": offset_err,
            "Precision": precision, "Recall": recall, "Specificity": specificity, "BalAcc": bal_acc,
            "F1": base_f1, "F1_95CI_Lower": f1_ci[0], "F1_95CI_Upper": f1_ci[1],
            "MCC": base_mcc, "MCC_95CI_Lower": mcc_ci[0], "MCC_95CI_Upper": mcc_ci[1],
            "AUPRC": base_auprc, "Neg_AUPRC": base_neg_auprc,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })
        
        # 1. PR Curve Generation
        PrecisionRecallDisplay.from_predictions(inst_y_true, inst_probs, name=f"{inst_name} (AUPRC={base_auprc:.3f})")
        plt.title(f"Precision-Recall Curve: {inst_name}\n({exp_name} | Seed {seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"Fig_PR_Curve_{exp_name}_Seed{seed}_{inst_name}.png"), dpi=300)
        plt.close()
        
        # 2. Calibration Plot Generation
        CalibrationDisplay.from_predictions(inst_y_true, inst_probs, n_bins=10, name=inst_name)
        plt.title(f"Calibration Plot: {inst_name}\n({exp_name} | Seed {seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"Fig_Calibration_{exp_name}_Seed{seed}_{inst_name}.png"), dpi=300)
        plt.close()
        
        # 3. Confusion Matrix Generation
        ConfusionMatrixDisplay.from_predictions(inst_y_true, inst_y_pred, display_labels=["Absent", "Active"], cmap="Blues")
        plt.title(f"Confusion Matrix: {inst_name}\n({exp_name} | Seed {seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"Fig_ConfMatrix_{exp_name}_Seed{seed}_{inst_name}.png"), dpi=300)
        plt.close()
        
    return pd.DataFrame(rows)

# --- DATASET & AUGMENTATIONS ---
class LogMelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=2048, hop_length=160, n_frames=501):
        self.n_frames = n_frames
        self.transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        mel = self.amplitude_to_db(self.transform(waveform))
        return F.pad(mel, (0, max(0, self.n_frames - mel.shape[-1])))[:, :, :self.n_frames]

def augment(waveform, sample_rate):
    if random.random() < 0.5:
        rate = random.uniform(0.85, 1.15)
        new_length = int(waveform.shape[1] / rate)
        waveform = F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
    if random.random() < 0.3:
        waveform += 0.005 * random.random() * torch.randn_like(waveform)
    if random.random() < 0.3:
        waveform *= (10 ** (random.uniform(-6, 6)/20))
    return waveform

def spec_augment(mel_spec):
    mel_spec = mel_spec.clone()
    f0 = random.randint(0, mel_spec.shape[-2] - 20)
    mel_spec[:, f0:f0+20, :] = 0
    t0 = random.randint(0, mel_spec.shape[-1] - 30)
    mel_spec[:, :, t0:t0+30] = 0
    return mel_spec

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, negative_prob=0.0, augment_flag=True):
        self.split = split
        self.transform = transform
        self.negative_prob = negative_prob
        self.augment_flag = augment_flag
        self.root_dir = os.path.join(root_dir, split)
        self.audio_paths = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.npy')])

    def __len__(self): return len(self.audio_paths)

    def instrument_presence_mask(self, stems_slice):
        dynamic_threshold = np.mean(stems_slice[0] ** 2) * 0.03 + 1e-9
        return torch.tensor([1.0 if np.mean(stems_slice[idx] ** 2) > dynamic_threshold else 0.0 for idx in range(1, 5)], dtype=torch.float32)

    def __getitem__(self, idx):
        stems = np.load(self.audio_paths[idx], mmap_mode='r')
        if self.split == 'test': return torch.tensor(np.array(stems)).float()
        
        start_idx = np.random.randint(0, stems.shape[1] - max_len) if stems.shape[1] > max_len else 0
        stems_slice = np.array(stems[:, start_idx : start_idx + max_len])
        if stems.shape[1] <= max_len: stems_slice = np.pad(stems_slice, ((0,0), (0, max_len - stems.shape[1])))

        if self.augment_flag and self.negative_prob > 0:
            for i in range(1, 5): 
                if random.random() < self.negative_prob: stems_slice[i] = 0.0
            stems_slice[0] = np.sum(stems_slice[1:], axis=0)

        waveform = torch.tensor(stems_slice[0]).float().unsqueeze(0)
        if self.augment_flag and torch.sum(torch.abs(waveform)) > 0: waveform = augment(waveform, 16000)

        if self.transform:
            waveform = self.transform(waveform)
            if torch.std(waveform) > 0: waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
            if waveform.dim() == 2: waveform = waveform.unsqueeze(0)
            if self.augment_flag: waveform = spec_augment(waveform)

        return waveform, self.instrument_presence_mask(stems_slice)

# --- ARCHITECTURE: PANNs CNN14 ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return F.avg_pool2d(x, kernel_size=2, stride=2)

class PANNs_CNN14(nn.Module):
    def __init__(self, num_classes=4):
        super(PANNs_CNN14, self).__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.classifier = nn.Linear(2048, num_classes)
        
    def load_audioset_weights(self):
        url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
        cache_path = "panns_cnn14_audioset.pth"
        if not os.path.exists(cache_path):
            console_log("Downloading official PANNs CNN14 AudioSet weights (~160MB)...")
            urllib.request.urlretrieve(url, cache_path)
            
        checkpoint = torch.load(cache_path, map_location='cpu', weights_only=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict and "classifier" not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        console_log("PANNs AudioSet weights successfully loaded!")

    def forward(self, x):
        # Align Torchaudio shape [Mels, Time] with PANNs shape [Time, Mels]
        x = x.transpose(2, 3)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0]
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.classifier(x)

def asymmetric_loss(output, target):
    x_sig = torch.sigmoid(output)
    return -torch.mean(target * torch.log(x_sig + 1e-8) + (1 - target) * torch.log(1 - x_sig + 1e-8) * x_sig)

def evaluate_val(model, dataloader, device):
    model.eval()
    y_true, probs = [], []
    with torch.no_grad():
        for mix, label in dataloader:
            y_true.append(label.numpy())
            probs.append(torch.sigmoid(model(mix.to(device))).cpu().numpy())
    y_true, probs = np.vstack(y_true), np.vstack(probs)
    
    thresholds = []
    for i in range(4):
        p, r, t = precision_recall_curve(y_true[:, i], probs[:, i])
        f1s = 2 * (p * r) / (p + r + 1e-9)
        thresholds.append(t[np.argmax(f1s[:-1])] if len(t) > np.argmax(f1s[:-1]) else 0.5)
        
    y_pred = (probs > np.array(thresholds)).astype(float)
    return f1_score(y_true, y_pred, average="macro", zero_division=0), np.array(thresholds)

if __name__ == "__main__":
    console_log(f"\n{'='*70}\n[START] PANNS CNN14 AUDIO-PRETRAINED BASELINE\n{'='*70}")
    transform = LogMelSpectrogramTransform()
    
    for seed in seeds:
        exp_name = "PANNs_CNN14_Baseline"
        run_key = f"{exp_name}_Seed{seed}"
        
        if os.path.exists(csv_path) and ((pd.read_csv(csv_path)['Experiment'] == exp_name) & (pd.read_csv(csv_path)['Seed'] == seed)).any():
            console_log(f"[SKIP] {run_key} is already fully evaluated.")
            continue
            
        console_log(f"\n--- Running for Seed: {seed} ---")
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        
        dataset_train = MUSDB18Dataset("musdb18_npy", 'train', transform, 0.3, True)
        train_idx, val_idx = train_test_split(list(range(len(dataset_train))), test_size=0.2, random_state=42)
        train_loader = DataLoader(Subset(dataset_train, train_idx), batch_size=batch_size, shuffle=True)
        val_loader_clean = DataLoader(Subset(MUSDB18Dataset("musdb18_npy", 'train', transform, 0, False), val_idx), batch_size=batch_size, shuffle=False)
        test_dataset = MUSDB18Dataset("musdb18_npy", 'test', None, 0, False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        model = PANNs_CNN14(num_classes=4).to(device)
        model.load_audioset_weights()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        save_path, checkpoint_path = os.path.join(results_dir, f"model_{run_key}.pt"), os.path.join(results_dir, f"checkpoint_{run_key}.pt")
        start_epoch, best_val, patience = 0, 0, 0
        
        if os.path.exists(checkpoint_path):
            console_log(f"[RESUME] Restoring from {checkpoint_path}...")
            cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(cp['model_state_dict']); optimizer.load_state_dict(cp['optimizer_state_dict']); scheduler.load_state_dict(cp['scheduler_state_dict'])
            start_epoch, best_val, patience = cp['epoch'] + 1, cp['best_val_macro_f1'], cp['patience_counter']
        
        for epoch in range(start_epoch, epochs):
            if patience >= early_stopping_patience: break
            model.train()
            running_loss = 0.0
            if epoch < warmup_epochs: 
                for pg in optimizer.param_groups: pg['lr'] = 1e-4 * min(1.0, float(epoch + 1) / warmup_epochs)
                    
            for mix, label in train_loader:
                optimizer.zero_grad()
                loss = asymmetric_loss(model(mix.to(device)), label.to(device))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm); optimizer.step()
                running_loss += loss.item()
            
            if epoch >= warmup_epochs: scheduler.step()
            val_f1, thresholds = evaluate_val(model, val_loader_clean, device)
            console_log(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val: best_val, patience = val_f1, 0; torch.save(model.state_dict(), save_path)
            else: patience += 1
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_macro_f1': best_val, 'patience_counter': patience}, checkpoint_path)
        
        if os.path.exists(save_path): model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        
        console_log("Running Final Test Evaluation...")
        model.eval(); y_true, probs_agg, track_ids = [], [], []
        nn_start_time = time.perf_counter()
        with torch.no_grad():
            for track_idx, full_stems in enumerate(test_loader): 
                for start in range(0, full_stems.shape[-1] - max_len + 1, max_len // 2):
                    stem = full_stems[0, :, start:start+max_len].numpy()
                    label = test_dataset.instrument_presence_mask(stem)
                    mel = transform(torch.tensor(stem[0]).float().unsqueeze(0))
                    if torch.std(mel) > 0: mel = (mel - mel.mean()) / (mel.std() + 1e-9)
                    probs_agg.append(torch.sigmoid(model(mel.unsqueeze(0).to(device))).cpu().numpy()[0])
                    y_true.append(label.numpy()); track_ids.append(track_idx)
        nn_ms = ((time.perf_counter() - nn_start_time) / len(probs_agg)) * 1000
        y_true, probs_agg, track_ids = np.vstack(y_true), np.vstack(probs_agg), np.array(track_ids)
        y_pred_raw = (probs_agg > thresholds).astype(float)
        
        dsp_start_time = time.perf_counter()
        y_pred_dsp = causal_dsp_post_processing(probs_agg, thresholds, 5)
        dsp_ms = ((time.perf_counter() - dsp_start_time) / len(probs_agg)) * 1000
        
        df_metrics = track_level_bootstrap_metrics(y_true, y_pred_raw, y_pred_dsp, probs_agg, track_ids, exp_name, seed, 2.5)
        console_log(f"[{exp_name}] FINAL METRICS (Seed {seed}) | NN Infer: {nn_ms:.2f}ms | DSP: {dsp_ms:.3f}ms")
        
        df_metrics.insert(0, "Seed", seed); df_metrics.insert(1, "Experiment", exp_name)
        df_metrics["NN_Infer_ms"], df_metrics["DSP_Overhead_ms"] = nn_ms, dsp_ms
        df_metrics.to_csv(csv_path, mode='a', header=not os.path.isfile(csv_path), index=False)