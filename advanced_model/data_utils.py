import os
import json
import random
import numpy as np

def split_data(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits samples into Train, Val, and Test sets.
    """
    random.seed(seed)
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    return train_samples, val_samples, test_samples

def get_sl_samples(tensor_root, label_map_path):
    """
    Returns list of (tensor_path, sentence_ids)
    """
    with open(label_map_path, 'r') as f:
        data = json.load(f)
        mapping = data['mapping']
    
    samples = []
    for sentence_class, token_ids in mapping.items():
        if not token_ids: continue
        
        class_dir = os.path.join(tensor_root, sentence_class)
        if not os.path.exists(class_dir):
            found = False
            if os.path.exists(tensor_root):
                for d in os.listdir(tensor_root):
                    if d.lower() == sentence_class.lower():
                        class_dir = os.path.join(tensor_root, d)
                        found = True
                        break
            if not found: continue
            
        if os.path.isdir(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            for f in files:
                samples.append((os.path.join(class_dir, f), token_ids))
    return samples
