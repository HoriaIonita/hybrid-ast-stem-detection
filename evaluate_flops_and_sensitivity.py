# -*- coding: utf-8 -*-
"""
Title: MDPI Rebuttal - FLOPs Profiler & Threshold Sensitivity Sweep
Description: 
Calculates computational overhead for AST vs PANNs CNN14.
Generates an F1 sensitivity curve across different ground-truth power thresholds.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from thop import profile

# Import architectures directly from your completed scripts
from proposed_ast_revised import TransferAST, MUSDB18Dataset, LogMelSpectrogramTransform, all_instruments, max_len
from panns_enhanced_dsp import PANNs_CNN14

results_dir = "Results_Rebuttal_Extras"
os.makedirs(results_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def profile_model(model, name, input_shape):
    model.eval().to(device)
    dummy_input = torch.randn(*input_shape).to(device)
    
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops_g = (macs * 2) / 1e9  # 1 MAC = 2 FLOPs
    params_m = params / 1e6
    
    print(f"[{name}] Parameters: {params_m:.2f} M | Compute: {flops_g:.2f} GFLOPs")
    return params_m, flops_g

def run_threshold_sweep():
    print(f"\n--- Running Ground-Truth Power Threshold Sweep (0.01 to 0.06) ---")
    transform = LogMelSpectrogramTransform()
    test_dataset = MUSDB18Dataset("musdb18_npy", split='test', transform=None, augment_flag=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load the best performing AST model from Seed 42
    model = TransferAST(num_classes=4).to(device)
    model_path = "Results_Proposed_AST/model_Proposed_AST_Revision_Seed42.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Ensure the AST script has finished.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    sweep_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    hop_length = max_len // 2
    raw_probs, all_stems = [], []
    
    # 1. Run network inference exactly once
    print("Extracting Neural Network probabilities...")
    with torch.no_grad():
        for full_stems in test_loader:
            total_samples = full_stems.shape[-1]
            for start in range(0, total_samples - max_len + 1, hop_length):
                stem_segment = full_stems[0, :, start:start+max_len].numpy()
                all_stems.append(stem_segment)
                
                waveform = torch.tensor(stem_segment[0]).float().unsqueeze(0)
                mel = transform(waveform)
                if torch.std(mel) > 0: mel = (mel - mel.mean()) / (mel.std() + 1e-9)
                
                out = model(mel.unsqueeze(0).to(device))
                raw_probs.append(torch.sigmoid(out).cpu().numpy()[0])
                
    raw_probs = np.vstack(raw_probs)
    
    # Standard decision threshold for evaluation 
    # (Since we are testing ground-truth sensitivity, we freeze the PR-curve threshold logic to a baseline)
    eval_preds = (raw_probs > 0.15).astype(float) 
    
    # 2. Sweep the dataset generation logic
    f1_results = {inst: [] for inst in all_instruments}
    
    for thresh in sweep_values:
        print(f"Generating labels and evaluating F1 for Power Threshold: {thresh}...")
        y_true_sweep = []
        for stem_seg in all_stems:
            mixture_power = np.mean(stem_seg[0] ** 2)
            dynamic_thresh = mixture_power * thresh + 1e-9
            presence_flags = [1.0 if np.mean(stem_seg[idx] ** 2) > dynamic_thresh else 0.0 for idx in range(1, 5)]
            y_true_sweep.append(presence_flags)
            
        y_true_sweep = np.vstack(y_true_sweep)
        
        for i, inst in enumerate(all_instruments):
            f1 = f1_score(y_true_sweep[:, i], eval_preds[:, i], zero_division=0)
            f1_results[inst].append(f1)
            
    # 3. Plot the Sensitivity Curve
    plt.figure(figsize=(8, 5))
    for inst in all_instruments:
        plt.plot(sweep_values, f1_results[inst], marker='o', label=inst.capitalize())
        
    plt.axvline(x=0.03, color='r', linestyle='--', label='Chosen Baseline (0.03)')
    plt.title("AST Performance vs Ground-Truth Power Threshold")
    plt.xlabel("Power Threshold Ratio")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Fig_Threshold_Sensitivity_Sweep_Power.png"), dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Sensitivity Sweep Plot saved to {results_dir}/Fig_Threshold_Sensitivity_Sweep.png")

if __name__ == "__main__":
    print(f"\n{'='*70}\n[START] ARCHITECTURE PROFILING & SENSITIVITY SWEEP\n{'='*70}")
    
    print("\n--- 1. Computational Complexity (FLOPs & Parameters) ---")
    
    # AST uses 128 Mel bins
    ast_input_shape = (1, 1, 128, 501)
    ast_model = TransferAST(num_classes=4)
    profile_model(ast_model, "Proposed AST", ast_input_shape)
    
    # PANNs CNN14 uses 64 Mel bins
    cnn_input_shape = (1, 1, 64, 501)
    cnn_model = PANNs_CNN14(num_classes=4)
    profile_model(cnn_model, "Baseline CNN14", cnn_input_shape)
    
    # Free up memory
    del ast_model
    del cnn_model
    torch.cuda.empty_cache()
    
    # Run the sweep
    run_threshold_sweep()
