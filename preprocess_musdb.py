# -*- coding: utf-8 -*-
"""
Title: Transfer-Learning Audio Spectrogram Transformer for Multi-Label Stem Detection
Authors: Horia-Sebastian Ioniță

Description:
Data preprocessing script for the MUSDB18 dataset.
Solves CPU/Disk I/O bottlenecks by converting heavy .stem.mp4 files
into lightweight, 16kHz mono memory-mapped .npy arrays.
"""

import os
import numpy as np
import stempeg
from tqdm import tqdm

def preprocess_musdb(source_dir="musdb18", target_dir="musdb18_npy", sample_rate=16000):
    os.makedirs(target_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        split_source = os.path.join(source_dir, split)
        split_target = os.path.join(target_dir, split)
        os.makedirs(split_target, exist_ok=True)
        
        audio_files = [f for f in os.listdir(split_source) if f.endswith('.stem.mp4')]
        print(f"\nProcessing {split} set ({len(audio_files)} files)...")
        
        for f in tqdm(audio_files):
            source_path = os.path.join(split_source, f)
            target_path = os.path.join(split_target, f.replace('.stem.mp4', '.npy'))
            
            # Skip if already processed
            if os.path.exists(target_path):
                continue
                
            # Decode the MP4
            stems, rate = stempeg.read_stems(source_path, sample_rate=sample_rate)
            
            # Convert to Mono to save space and I/O time
            stems_mono = np.mean(stems, axis=2).astype(np.float32)
            
            # Save as uncompressed numpy array for instant memory-mapping
            np.save(target_path, stems_mono)

if __name__ == "__main__":
    preprocess_musdb()
    print("\nPreprocessing complete. You can now use 'musdb18_npy' for lightning-fast training.")