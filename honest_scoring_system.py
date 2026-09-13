# -*- coding: utf-8 -*-
"""
Title: Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection
Authors: Horia-Sebastian Ioniță

Description:
Honest Evaluation and Classification System for All Models.
1. Penalizes standard deviation (instability).
2. Eliminates redundant/deceptive metrics.
3. Applies academic weights (MCC > BalAcc > F1).
4. Generates graphs (Final Bar Chart, Stability Scatter, Radar Chart).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from math import pi

# --- RAW DATA ---
raw_data = """Base_Experiment	Instrument	F1	Precision	Recall	Specificity	FPR	BalAcc	MCC	Neg_AUPRC	TP	FP	FN	TN
Ablation_NoAugment	MACRO_AVG	0.941 ± 0.007	0.931 ± 0.027	0.954 ± 0.021	0.541 ± 0.167	0.459 ± 0.167	0.748 ± 0.074	0.544 ± 0.077	0.728 ± 0.046	15753 ± 336	1188 ± 496	719 ± 336	1976 ± 496
Ablation_NoOverlap	MACRO_AVG	0.921 ± 0.001	0.928 ± 0.036	0.919 ± 0.034	0.596 ± 0.203	0.404 ± 0.203	0.758 ± 0.084	0.523 ± 0.086	0.676 ± 0.060	7626 ± 260	599 ± 339	636 ± 260	1015 ± 339
Ablation_NoPretrain	MACRO_AVG	0.889 ± 0.016	0.874 ± 0.011	0.909 ± 0.040	0.339 ± 0.117	0.661 ± 0.117	0.624 ± 0.039	0.293 ± 0.047	0.435 ± 0.039	15084 ± 652	2084 ± 322	1388 ± 652	1080 ± 322
CNN_Enhanced_DSP	MACRO_AVG	0.897 ± 0.016	0.858 ± 0.005	0.946 ± 0.039	0.196 ± 0.046	0.804 ± 0.046	0.571 ± 0.021	0.188 ± 0.071	0.304 ± 0.041	15603 ± 566	2572 ± 269	869 ± 566	592 ± 269
ResNet18_Enhanced_DSP	MACRO_AVG	0.903 ± 0.014	0.865 ± 0.021	0.948 ± 0.004	0.246 ± 0.152	0.754 ± 0.152	0.597 ± 0.078	0.226 ± 0.178	0.338 ± 0.150	15638 ± 71	2406 ± 456	834 ± 71	758 ± 456
Source_Drop_0.7	MACRO_AVG	0.921 ± 0.007	0.923 ± 0.018	0.922 ± 0.023	0.593 ± 0.191	0.407 ± 0.191	0.758 ± 0.086	0.512 ± 0.102	0.661 ± 0.085	15183 ± 422	1258 ± 340	1289 ± 422	1906 ± 340
Proposed_AST_DSP_Hysteresis	MACRO_AVG	0.922 ± 0.017	0.906 ± 0.012	0.943 ± 0.023	0.479 ± 0.017	0.521 ± 0.017	0.711 ± 0.010	0.469 ± 0.046	0.683 ± 0.034	15532 ± 397	1588 ± 219	940 ± 397	1576 ± 219
Proposed_AST_DSP_w7	MACRO_AVG	0.926 ± 0.022	0.931 ± 0.007	0.925 ± 0.047	0.630 ± 0.110	0.370 ± 0.110	0.778 ± 0.034	0.546 ± 0.039	0.683 ± 0.034	15238 ± 805	1104 ± 158	1234 ± 805	2060 ± 158
Initial_Proposed_AST	MACRO_AVG	0.914 ± 0.009	0.934 ± 0.035	0.901 ± 0.043	0.637 ± 0.197	0.363 ± 0.197	0.769 ± 0.080	0.527 ± 0.083	0.666 ± 0.059	14881 ± 688	1059 ± 669	1591 ± 688	2105 ± 669
Source_Drop_0.0	MACRO_AVG	0.934 ± 0.004	0.929 ± 0.030	0.943 ± 0.030	0.551 ± 0.215	0.449 ± 0.215	0.747 ± 0.094	0.522 ± 0.125	0.697 ± 0.066	15568 ± 476	1237 ± 582	904 ± 476	1927 ± 582
Source_Drop_0.3	MACRO_AVG	0.921 ± 0.015	0.939 ± 0.019	0.908 ± 0.043	0.683 ± 0.144	0.317 ± 0.144	0.796 ± 0.052	0.556 ± 0.044	0.682 ± 0.044	14945 ± 730	964 ± 355	1527 ± 730	2200 ± 355
Loss_BCE	MACRO_AVG	0.914 ± 0.019	0.927 ± 0.028	0.909 ± 0.041	0.616 ± 0.173	0.384 ± 0.173	0.762 ± 0.072	0.520 ± 0.071	0.679 ± 0.046	14978 ± 689	1178 ± 549	1494 ± 689	1986 ± 549
Loss_ASL_Gamma_1.0	MACRO_AVG	0.928 ± 0.011	0.935 ± 0.031	0.926 ± 0.025	0.650 ± 0.179	0.350 ± 0.179	0.788 ± 0.080	0.573 ± 0.086	0.706 ± 0.052	15274 ± 407	1060 ± 566	1198 ± 407	2104 ± 566
Loss_ASL_Gamma_3.0	MACRO_AVG	0.922 ± 0.009	0.934 ± 0.033	0.915 ± 0.036	0.660 ± 0.207	0.340 ± 0.207	0.788 ± 0.087	0.556 ± 0.087	0.684 ± 0.055	15097 ± 574	1055 ± 623	1375 ± 574	2109 ± 623
Proposed_AST_DSP	MACRO_AVG	0.923 ± 0.006	0.962 ± 0.012	0.892 ± 0.021	0.778 ± 0.058	0.222 ± 0.058	0.835 ± 0.019	0.614 ± 0.023	0.755 ± 0.007	14753 ± 290	588 ± 203	1719 ± 290	2576 ± 203
Proposed_AST	MACRO_AVG	0.930 ± 0.002	0.949 ± 0.021	0.916 ± 0.025	0.744 ± 0.119	0.256 ± 0.119	0.830 ± 0.047	0.609 ± 0.027	0.755 ± 0.007	15079 ± 407	819 ± 374	1393 ± 407	2345 ± 374
"""

# --- PARSING AND COMPUTATION ---
def parse_and_score(raw_text):
    df_raw = pd.read_csv(io.StringIO(raw_text), sep='\t')
    
    # 1. Extract Mean and Std
    metrics = ['F1', 'Specificity', 'BalAcc', 'MCC', 'Neg_AUPRC']
    parsed_data = []
    
    for _, row in df_raw.iterrows():
        model_name = row['Base_Experiment']
        model_data = {'Model': model_name}
        
        for m in metrics:
            val_str = str(row[m]).strip()
            if '±' in val_str:
                mean_val, std_val = val_str.split('±')
                model_data[f'{m}_mean'] = float(mean_val.strip())
                model_data[f'{m}_std'] = float(std_val.strip())
            else:
                model_data[f'{m}_mean'] = float(val_str)
                model_data[f'{m}_std'] = 0.0
                
        parsed_data.append(model_data)
        
    df = pd.DataFrame(parsed_data)
    
    # 2. Honest Formula: Score = Mean - Std (Instability penalty)
    for m in metrics:
        df[f'{m}_score'] = df[f'{m}_mean'] - df[f'{m}_std']
        
    # 3. Min-Max Scaling (normalize between 0 and 1 for each score column)
    for m in metrics:
        min_val = df[f'{m}_score'].min()
        max_val = df[f'{m}_score'].max()
        if max_val > min_val:
            df[f'{m}_norm'] = (df[f'{m}_score'] - min_val) / (max_val - min_val)
        else:
            df[f'{m}_norm'] = 0.0
            
    # 4. Academic Weights (Total = 1.0)
    weights = {
        'MCC_norm': 0.35,        # Global capacity of imbalanced model
        'BalAcc_norm': 0.25,     # Balance between TPR and TNR
        'Specificity_norm': 0.20,# Capacity to reject noise (essential)
        'Neg_AUPRC_norm': 0.10,  # Precision on negative class (silence)
        'F1_norm': 0.10          # F1-Score (low weight because it hides False Positives)
    }
    
    df['Honest_Score_100'] = 0
    for w_key, w_val in weights.items():
        df['Honest_Score_100'] += df[w_key] * w_val
        
    df['Honest_Score_100'] = (df['Honest_Score_100'] * 100).round(2)
    
    return df.sort_values(by='Honest_Score_100', ascending=False).reset_index(drop=True)

# --- GRAPHICS ---
def generate_graphs(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # GRAPH 1: Final Ranking (Bar Chart)
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Honest_Score_100', y='Model', data=df, palette='viridis')
    plt.title('Final Honest Ranking (Score 0–100) \nPenalized by Standard Deviation & Weighted MCC/BalAcc', fontsize=14, fontweight='bold')
    plt.xlabel('Honest Score')
    plt.ylabel('')
    for i, score in enumerate(df['Honest_Score_100']):
        plt.text(score + 0.5, i, f'{score:.2f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_Honest_Ranking.png'), dpi=300)
    plt.close()
    
    # GRAPH 2: Stability vs Performance (Scatter on MCC)
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='MCC_std', 
        y='MCC_mean', 
        hue='Model', 
        s=300, 
        data=df, 
        palette='tab20', 
        legend=True
    )
    plt.title('Stability Analysis: Performance vs Variation (MCC)', fontsize=14, fontweight='bold')
    plt.xlabel('Instability (MCC Standard Deviation) -> Smaller is better')
    plt.ylabel('Performance (MCC Average) -> Greater is better')
    plt.gca().invert_xaxis()
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        borderaxespad=0., 
        title="Evaluated Models",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_Stability_vs_Performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # GRAPH 3: Radar Chart (Top 3 Models)
    categories = ['F1', 'Specificity', 'BalAcc', 'MCC', 'Neg_AUPRC']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)
    
    colors = ['b', 'g', 'r']
    for i in range(3): # Plot top 3
        values = df.iloc[i][[f'{c}_norm' for c in categories]].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=df.iloc[i]['Model'], color=colors[i])
        ax.fill(angles, values, colors[i], alpha=0.1)
        
    plt.title('Metrics Equilibrium (Top 3 Models)', size=15, color='black', y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(os.path.join(output_dir, '3_Radar_Chart_Top3.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    out_dir = "Classification_Results"
    df_ranked = parse_and_score(raw_data)
    generate_graphs(df_ranked, out_dir)
    
    print("\n" + "="*80)
    print(" Final Ranking (Penalized for Instability)")
    print("="*80)
    for i, row in df_ranked.iterrows():
        print(f"{i+1:2d}. {row['Model']:<40} | Score: {row['Honest_Score_100']:>6.2f}/100")
    print("="*80)
    print(f"\n[SUCCESS] Graphs successfully saved to folder: '{out_dir}'")