import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, data_root, max_frames=150, padding_mode='zero', augment=False):
        """
        Args:
            data_root (str): Path to Videos_tensors directory
            max_frames (int): Target length for padding/cropping
            padding_mode (str): 'zero' or 'repeat'
            augment (bool): Whether to apply data augmentation
        """
        self.data_root = data_root
        self.max_frames = max_frames
        self.padding_mode = padding_mode
        self.augment = augment
        
        self.samples = []
        self.classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
        # Scan files
        for cls_name in self.classes:
            cls_dir = os.path.join(data_root, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.npy'):
                    self.samples.append({
                        'path': os.path.join(cls_dir, file_name),
                        'label': self.class_to_idx[cls_name]
                    })
        
        print(f"Dataset Loaded: {len(self.samples)} samples from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load Data: (T, V, C) where C=3 (x, y, z)
        # Original shape from final_preprocessing.py is (T, V, 3)
        data = np.load(item['path'])
        
        # Handle Empty or Broken files
        if data.shape[0] == 0:
            # Return a zero tensor if empty (should check during scanning ideally)
            data = np.zeros((1, 119, 3), dtype=np.float32)

        T, V, C = data.shape
        
        # 1. Processing / Normalization
        # Center the skeleton: Subtract the first frame's Nose (Index 0) from all frames/joints
        # This makes the data invariant to where the person is standing in the room.
        origin = data[0, 0, :] # (3,) - Nose of first frame
        data = data - origin # Broadcasting subtract
        
        # 2. Reshape to (C, T, V, M) for ST-GCN
        # Current: (T, V, C) -> Transpose -> (C, T, V)
        data = data.transpose(2, 0, 1) # (C, T, V)
        
        # Add Person Dimension (M=1) -> (C, T, V, M)
        data = data[:, :, :, np.newaxis] # (C, T, V, 1)

        # 3. Padding / Cropping
        C, T, V, M = data.shape
        
        if T < self.max_frames:
            # Pad
            if self.padding_mode == 'zero':
                pad_len = self.max_frames - T
                # Pad along T dimension (axis 1)
                padding = np.zeros((C, pad_len, V, M), dtype=data.dtype)
                data = np.concatenate((data, padding), axis=1)
            elif self.padding_mode == 'loop':
                # Repeat sequence
                tile_count = (self.max_frames // T) + 1
                data = np.tile(data, (1, tile_count, 1, 1))[:, :self.max_frames, :, :]
        elif T > self.max_frames:
            # Crop (Center or Random? Let's do Center for validation, Random for train?)
            # For simplicity, just take the first max_frames
            data = data[:, :self.max_frames, :, :]
            
        # Convert to Tensor
        data_tensor = torch.from_numpy(data).float()
        
        # 4. Augmentation (Only if enabled)
        if self.augment:
             # Random Scale (0.9 to 1.1)
             scale = 0.9 + (torch.rand(1).item() * 0.2)
             data_tensor = data_tensor * scale

        label_tensor = torch.tensor(item['label'], dtype=torch.long)
        
        return data_tensor, label_tensor

if __name__ == "__main__":
    # Test Block
    ROOT = r"e:/projects/sign language/Videos_tensors"
    dataset = SignLanguageDataset(ROOT, max_frames=100)
    
    if len(dataset) > 0:
        data, label = dataset[0]
        print(f"Sample 0 Shape: {data.shape}") # Should be (3, 100, 119, 1)
        print(f"Sample 0 Label: {label} ({dataset.idx_to_class[label.item()]})")
        
        # Verify Batch
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        for batch_data, batch_labels in dataloader:
            print(f"Batch Shape: {batch_data.shape}")
            print(f"Batch Labels: {batch_labels}")
            break
    else:
        print("No samples found. Check path.")
