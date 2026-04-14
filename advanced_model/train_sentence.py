import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from tqdm import tqdm
from st_transformer import SignTransformer

# CONFIGURATION
SENTENCE_TENSOR_ROOT = r"e:/projects/sign language/Videos_tensors"
MAP_FILE = r"e:/projects/sign language/advanced_model/sentence_map.json"
CHECKPOINT_DIR = r"e:/projects/sign language/advanced_model/checkpoints_sentence"
RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sentence_model_best.pth")
PRETRAIN_CHECKPOINT = r"e:/projects/sign language/advanced_model/checkpoints_word/word_model_epoch_50.pth"
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 200 # Total target epochs
MAX_FRAMES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class SentenceDataset(Dataset):
    def __init__(self, data_root, map_file, max_frames=100, augment=True, samples=None):
        self.data_root = data_root
        self.max_frames = max_frames
        self.augment = augment
        
        with open(map_file, 'r') as f:
            data = json.load(f)
            self.mapping = data['mapping']
            self.vocab = data['vocab']
            
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            for sentence_class in self.mapping:
                token_ids = self.mapping[sentence_class]
                if not token_ids: continue
                    
                class_dir = os.path.join(data_root, sentence_class)
                if not os.path.exists(class_dir):
                    found = False
                    for d in os.listdir(data_root):
                        if d.lower() == sentence_class.lower():
                            class_dir = os.path.join(data_root, d)
                            found = True
                            break
                    if not found: continue
    
                files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
                for f in files:
                    self.samples.append((os.path.join(class_dir, f), token_ids))
        print(f"Loaded {len(self.samples)} Sentence Samples.")

    def __len__(self):
        return len(self.samples)

    def apply_augmentation(self, data):
        T, V, C = data.shape
        if np.random.rand() > 0.5:
            data = data * np.random.uniform(0.9, 1.1)
        if np.random.rand() > 0.5:
            theta = np.radians(np.random.uniform(-10, 10))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            data = np.dot(data.reshape(-1, 3), rotation_matrix).reshape(T, V, 3)
        if np.random.rand() > 0.5:
            data = data + np.random.normal(0, 0.005, data.shape)
        return data

    def __getitem__(self, idx):
        path, token_ids = self.samples[idx]
        data = np.load(path)
        if self.augment:
            data = self.apply_augmentation(data)
        if data.shape[0] > 0:
            data = data - data[0, 0, :]
        T, V, C = data.shape
        data = data.transpose(2, 0, 1)[:, :, :, np.newaxis]
        if T < self.max_frames:
            pad = np.zeros((C, self.max_frames - T, V, 1))
            data = np.concatenate([data, pad], axis=1)
        elif T > self.max_frames:
            data = data[:, :self.max_frames, :, :]
        return torch.from_numpy(data).float(), torch.tensor(token_ids, dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = torch.cat(targets)
    return inputs, targets_flat, target_lengths

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, device='cuda', checkpoint_path='best_model.pth'):
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets_flat, target_lengths in pbar:
            inputs = inputs.to(device)
            targets_flat = (targets_flat + 1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            input_lengths = torch.full((inputs.size(0),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets_flat, target_lengths in val_loader:
                inputs = inputs.to(device)
                targets_flat = (targets_flat + 1).to(device)
                outputs = model(inputs)
                log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
                input_lengths = torch.full((inputs.size(0),), log_probs.size(0), dtype=torch.long)
                loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
    return best_loss

def main():
    from data_utils import get_sl_samples, split_data
    all_samples = get_sl_samples(SENTENCE_TENSOR_ROOT, MAP_FILE)
    train_s, val_s, test_s = split_data(all_samples)
    
    train_dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, augment=True, samples=train_s)
    val_dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, augment=False, samples=val_s)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    num_classes = len(train_dataset.vocab) + 1
    model = SignTransformer(num_classes=num_classes, phase='translation')
    
    if os.path.exists(RESUME_CHECKPOINT):
        model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=DEVICE))
    
    train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, checkpoint_path=RESUME_CHECKPOINT)

if __name__ == "__main__":
    main()
