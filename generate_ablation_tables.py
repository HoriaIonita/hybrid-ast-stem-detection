# -*- coding: utf-8 -*-
"""
Title: Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection
Authors: Horia-Sebastian Ioniță

Description:
Statistical Analysis script for the Ablation Study.
- Evaluates ALL classes (Drums, Bass, Other, Vocals).
- Calculates the MACRO average across all instruments.
- Generates comparative graphs (2x2 grids) and formatted CSV tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# --- CONFIGURATION ---
CSV_PATH = "Results_Ablation_Master/master_ablation_metrics.csv"
OUTPUT_DIR = "Statistical_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set graph styles
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14})

def clean_experiment_name(exp_string):
    """Removes the '_SeedXXX' suffix to group experiments."""
    return re.sub(r'_Seed\d+', '', str(exp_string))

def format_mean_std(mean, std, is_int=False):
    """Formats the result as 'Mean ± Std'"""
    if pd.isna(mean) or pd.isna(std):
        return "-"
    if is_int:
        return f"{int(round(mean))} ± {int(round(std))}"
    return f"{mean:.3f} ± {std:.3f}"

def run_analysis():
    print(f"Loading data from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Could not find file {CSV_PATH}.")
        return

    # 1. Clean Experiment Names
    df['Base_Experiment'] = df['Experiment'].apply(clean_experiment_name)
    
    float_metrics = ['F1', 'Precision', 'Recall', 'Specificity', 'FPR', 'BalAcc', 'MCC', 'Neg_AUPRC']
    int_metrics = ['TP', 'FP', 'FN', 'TN']
    all_metrics = float_metrics + int_metrics
    
    # 2. Create MACRO_AVG class (Average over all 4 instruments per run)
    macro_float = df.groupby(['Experiment', 'Base_Experiment', 'Seed'])[float_metrics].mean().reset_index()
    macro_int = df.groupby(['Experiment', 'Base_Experiment', 'Seed'])[int_metrics].sum().reset_index()
    
    macro_df = pd.merge(macro_float, macro_int, on=['Experiment', 'Base_Experiment', 'Seed'])
    macro_df['Instrument'] = 'MACRO_AVG'
    
    # Combine original data with Macro averages
    df_full = pd.concat([df, macro_df], ignore_index=True)
    
    # 3. Group and Statistical Calculation (Mean ± Std over 3 seeds)
    grouped = df_full.groupby(['Base_Experiment', 'Instrument'])[all_metrics].agg(['mean', 'std'])
    
    formatted_df = pd.DataFrame(index=grouped.index)
    
    for metric in float_metrics:
        formatted_df[metric] = grouped.apply(
            lambda row: format_mean_std(row[(metric, 'mean')], row[(metric, 'std')], is_int=False), axis=1
        )
    for metric in int_metrics:
        formatted_df[metric] = grouped.apply(
            lambda row: format_mean_std(row[(metric, 'mean')], row[(metric, 'std')], is_int=True), axis=1
        )
        
    # Save Full Table
    master_table_path = os.path.join(OUTPUT_DIR, "Master_Table_All_Classes.csv")
    formatted_df.reset_index().to_csv(master_table_path, index=False)
    print(f"\n[SUCCESS] Formatted table with all instruments saved to: {master_table_path}")
    
    # =========================================================================
    # 4. GENERATING MULTI-CLASS GRAPHS (2x2 Grids)
    # =========================================================================
    print("\nGenerating comparative graphs (All Instruments)...")
    
    instruments = ['drums', 'bass', 'other', 'vocals']
    order = ['Proposed_AST'] + sorted([e for e in df['Base_Experiment'].unique() if e != 'Proposed_AST'])

    # HELPER FUNCTION FOR 2x2 GRIDS (F1 and MCC)
    def plot_2x2_metric(metric_name, title_prefix, filename):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
        fig.suptitle(f'Comparison {metric_name} on All Instruments', fontsize=20, y=0.98)
        
        for i, inst in enumerate(instruments):
            ax = axes[i // 2, i % 2]
            inst_df = df[df['Instrument'] == inst]
            
            sns.barplot(data=inst_df, x=metric_name, y='Base_Experiment', order=order, 
                        ax=ax, ci='sd', palette='mako')
            
            ax.set_title(f'Instrument: {inst.upper()}', fontsize=16)
            ax.set_ylabel('')
            ax.set_xlabel(metric_name)
            
            # Add mean value to the bar
            means = inst_df.groupby('Base_Experiment')[metric_name].mean()
            for idx, exp_name in enumerate(order):
                if exp_name in means:
                    val = means[exp_name]
                    ax.text(val + 0.01, idx, f'{val:.3f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # Generate F1-Score Graph (2x2)
    plot_2x2_metric('F1', 'F1-Score', 'All_F1_Comparison.png')
    
    # Generate MCC Graph (2x2)
    plot_2x2_metric('MCC', 'Matthews Correlation Coef (MCC)', 'All_MCC_Comparison.png')

    # -------------------------------------------------------------------------
    # Graph 3: FP vs FN Trade-off (2x2 Grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Operational Trade-off: False Positives vs False Negatives', fontsize=20, y=0.95)

    for i, inst in enumerate(instruments):
        ax = axes[i // 2, i % 2]
        inst_df = df[df['Instrument'] == inst]
        
        # Calculate means on FP and FN for scatter plot
        mean_df = inst_df.groupby('Base_Experiment')[['FP', 'FN', 'F1']].mean().reset_index()
        
        sns.scatterplot(
            data=mean_df, x='FP', y='FN', hue='Base_Experiment', style='Base_Experiment',
            s=300, palette='tab10', ax=ax, edgecolor='black', linewidth=1
        )
        
        # Reference lines for SOTA
        if 'Proposed_AST' in mean_df['Base_Experiment'].values:
            sota_fp = mean_df[mean_df['Base_Experiment']=='SOTA']['FP'].values[0]
            sota_fn = mean_df[mean_df['Base_Experiment']=='SOTA']['FN'].values[0]
            ax.axvline(x=sota_fp, color='red', linestyle='--', alpha=0.5, label='Proposed Ref FP')
            ax.axhline(y=sota_fn, color='red', linestyle='--', alpha=0.5, label='Proposed Ref FN')
            
        ax.set_title(f'{inst.upper()}', fontsize=16)
        ax.set_xlabel('False Positives (Incorrect gate openings)')
        ax.set_ylabel('False Negatives (Missed instrument)')
        
        # Move legend outside the graph only for the last plot
        if i == 3:
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title='Experiments')
        else:
            ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'All_FP_vs_FN_Tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Graphs (2x2 Grids) successfully saved to folder: {OUTPUT_DIR}/")

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    run_analysis()