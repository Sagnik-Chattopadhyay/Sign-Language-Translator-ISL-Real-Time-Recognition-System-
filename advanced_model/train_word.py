import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from st_transformer import SignTransformer

# CONFIGURATION
DATA_ROOT = r"e:/projects/sign language/advanced_model/data_words"
CHECKPOINT_DIR = r"e:/projects/sign language/advanced_model/checkpoints_word"
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
MAX_FRAMES = 50 # Words are shorter than sentences
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class WordDataset(Dataset):
    def __init__(self, data_root, max_frames=50):
        self.data_root = data_root
        self.max_frames = max_frames
        self.samples = []
        self.classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Loading {len(self.classes)} Word Classes...")
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_root, cls_name)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
            for f in files:
                self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path) # (T, 119, 3)
        
        # Preprocessing
        # 1. Normalization (Centering)
        if data.shape[0] > 0:
            origin = data[0, 0, :]
            data = data - origin

        # 2. Reshape to (C, T, V, M)
        # Input: (T, V, C)
        T, V, C = data.shape
        data = data.transpose(2, 0, 1) # (C, T, V)
        data = data[:, :, :, np.newaxis] # (C, T, V, M)
        
        # 3. Padding/Cutting
        C, T, V, M = data.shape
        if T < self.max_frames:
            pad = np.zeros((C, self.max_frames - T, V, M))
            data = np.concatenate([data, pad], axis=1)
        elif T > self.max_frames:
            data = data[:, :self.max_frames, :, :]
            
        return torch.from_numpy(data).float(), label

def main():
    # 1. Dataset
    dataset = WordDataset(DATA_ROOT, max_frames=MAX_FRAMES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(dataset.classes)
    print(f"Dataset Size: {len(dataset)} samples. Classes: {num_classes}")
    
    # 2. Model
    model = SignTransformer(num_classes=num_classes, phase='pretrain')
    model.to(DEVICE)
    
    # 3. Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    print(f"Starting Training on {DEVICE}...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs) # (N, Num_Classes)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} Acc: {acc:.4f}")
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"word_model_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()
