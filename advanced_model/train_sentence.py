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
PRETRAIN_CHECKPOINT = r"e:/projects/sign language/advanced_model/checkpoints_word/word_model_epoch_50.pth" # Ideal
BATCH_SIZE = 4
LEARNING_RATE = 0.0001 # Slower for fine-tuning
NUM_EPOCHS = 100
MAX_FRAMES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class SentenceDataset(Dataset):
    def __init__(self, data_root, map_file, max_frames=100):
        self.data_root = data_root
        self.max_frames = max_frames
        
        # Load Map
        with open(map_file, 'r') as f:
            data = json.load(f)
            self.mapping = data['mapping'] # "sentence": [id1, id2]
            self.vocab = data['vocab']
            
        self.samples = []
        
        # Iterate over sentences
        for sentence_class in self.mapping:
            token_ids = self.mapping[sentence_class]
            if not token_ids: # Skip empty (all unknown) sentences
                continue
                
            class_dir = os.path.join(data_root, sentence_class)
            if not os.path.exists(class_dir):
                # Maybe folder name mismatch? "comb your hair" vs "Comb Your Hair"
                # Try simple variations
                found = False
                for d in os.listdir(data_root):
                    if d.lower() == sentence_class.lower():
                        class_dir = os.path.join(data_root, d)
                        found = True
                        break
                if not found:
                    continue

            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            for f in files:
                self.samples.append((os.path.join(class_dir, f), token_ids))
                
        print(f"Loaded {len(self.samples)} Sentence Samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, token_ids = self.samples[idx]
        data = np.load(path) # (T, 119, 3)
        
        # Preprocessing (Same as Word)
        if data.shape[0] > 0:
            origin = data[0, 0, :]
            data = data - origin

        # (T, V, C) -> (C, T, V, M)
        T, V, C = data.shape
        data = data.transpose(2, 0, 1) # (C, T, V)
        data = data[:, :, :, np.newaxis] # (C, T, V, M)
        
        # Padding
        C, T, V, M = data.shape
        if T < self.max_frames:
            pad = np.zeros((C, self.max_frames - T, V, M))
            data = np.concatenate([data, pad], axis=1)
        elif T > self.max_frames:
            data = data[:, :self.max_frames, :, :]
            
        return torch.from_numpy(data).float(), torch.tensor(token_ids, dtype=torch.long)

def collate_fn(batch):
    # Dynamic padding for inputs? No, fixed max_frames.
    # But TARGETS are variable length!
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    
    # Flatten targets for CTC Loss? Or pad?
    # CTC Loss needs (Target_Seq_Len,) per batch item.
    # Targets usually concatenated into 1D tensor for Pytorch CTCLoss.
    
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = torch.cat(targets)
    
    return inputs, targets_flat, target_lengths

def main():
    # 1. Dataset
    dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, max_frames=MAX_FRAMES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    num_classes = len(dataset.vocab) + 1 # +1 for CTC Blank
    print(f"Vocab Size: {len(dataset.vocab)}. CTC Classes: {num_classes}")
    
    # 2. Model
    # We use 'pretrain' mode logic because CTC effectively classifies frames
    # and then aligns them. A full Transformer Decoder is for Seq2Seq loss.
    # For simplicity in this iteration, we use CTC on the Encoder output.
    # If we want full Transformer, we need decoder inputs.
    # Let's stick to CTC on Encoder for now as it's closer to working.
    
    model = SignTransformer(num_classes=num_classes, phase='pretrain')
    
    # Load Pretrained Weights?
    if os.path.exists(PRETRAIN_CHECKPOINT):
        print(f"Loading Pretrained Weights: {PRETRAIN_CHECKPOINT}")
        # Be careful: Num classes might mismatch.
        # Word Model had 131 classes.
        # Sentence Model has 131 classes + 1 Blank.
        # We need to load carefully.
        state_dict = torch.load(PRETRAIN_CHECKPOINT)
        
        # Filter out classifier weights if shape mismatch
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and k != 'fc.weight' and k != 'fc.bias' and v.size() == model_dict[k].size()}
        
        # Explicitly skip 'fc' / 'classifier' if names differ or just to be safe for transfer learning
        # The 'fc' layer in stgcn.py is the classifier. In SignTransformer it uses 'classifier' or 'fc'?
        # st_transformer.py uses: self.encoder (STGCN) -> which has 'fc' (unused in encoder output)
        # AND self.classifier (Linear/Conv).
        
        # We need to check st_transformer keys.
        # It has 'classifier' (Linear) and 'encoder.model.fc' (Conv).
        # We want to keep encoder weights BUT NOT the classifier.
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers (Excluded classifier).")
    else:
        print("Warning: No Pretrained Checkpoint Found. Training from scratch.")

    model.to(DEVICE)
    
    # 3. Training
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # 0 is blank? Or last?
    # Usually we reserve 0 for blank.
    # Our vocab IDs are 0 to 130.
    # So we should shift labels + 1.
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    print(f"Starting Phase 2 Training (CTC) on {DEVICE}...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, targets_flat, target_lengths in pbar:
            inputs = inputs.to(DEVICE)
            targets_flat = targets_flat.to(DEVICE) + 1 # Shift for Blank=0
            
            optimizer.zero_grad()
            
            # Forward
            # Model outputs (N, Num_Classes) if aggregated.
            # But for CTC we need Time!
            # We need to modify st_transformer to return (N, T, C) if requested.
            # Currently 'pretrain' does global pooling.
            # We need a 'ctc' mode.
            
            # HACK: Use the encoder directly?
            # model.encoder returns (N, 256, T_out).
            # We need a projection (256 -> Num_Classes).
            # The 'classifier' is (256 -> Num_Classes).
            # So we can apply it frame-wise!
            
            features = model.encoder(inputs) # (N, 256, T_out)
            # Permute for linear layer: (N, T_out, 256)
            features = features.permute(0, 2, 1)
            
            # Project
            outputs = model.classifier(features) # (N, T_out, Num_Classes)
            
            # Prepare for CTC (T, N, C)
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            
            input_lengths = torch.full((inputs.size(0),), log_probs.size(0), dtype=torch.long)
            
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"sentence_model_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()
