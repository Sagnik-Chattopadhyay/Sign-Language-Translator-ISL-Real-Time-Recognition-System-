import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from train_sentence import SentenceDataset, collate_fn
from st_transformer import STGCN_Encoder
from baseline_models import (FCLSTM_Baseline, STGCN_LSTM_Baseline, 
                             STGCN_AttnLSTM_Baseline, STGCN_BiLSTM_Baseline, 
                             STGCN_GRU_Baseline, STGCN_BiGRU_Baseline,
                             STGCN_BiAttnLSTM_Baseline, CNN1D_LSTM_Baseline)

# CONFIG
SENTENCE_TENSOR_ROOT = r"e:/projects/sign language/Videos_tensors"
MAP_FILE = r"e:/projects/sign language/advanced_model/sentence_map.json"
CHECKPOINT_DIR = r"e:/projects/sign language/advanced_model/checkpoints_baselines"
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_FRAMES = 100
EPOCHS = 50 # Baselines train faster
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_model(model_name):
    # 1. Dataset
    dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, max_frames=MAX_FRAMES, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    num_classes = len(dataset.vocab) + 1
    
    # 2. Select Model
    if model_name == "fc-lstm":
        model = FCLSTM_Baseline(num_classes=num_classes)
    elif model_name == "stgcn-lstm":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_LSTM_Baseline(num_classes=num_classes, backbone=backbone)
    elif model_name == "stgcn-bilstm":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_BiLSTM_Baseline(num_classes=num_classes, backbone=backbone)
    elif model_name == "stgcn-gru":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_GRU_Baseline(num_classes=num_classes, backbone=backbone)
    elif model_name == "stgcn-bigru":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_BiGRU_Baseline(num_classes=num_classes, backbone=backbone)
    elif model_name == "stgcn-biattn":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_BiAttnLSTM_Baseline(num_classes=num_classes, backbone=backbone)
    elif model_name == "cnn1d-lstm":
        model = CNN1D_LSTM_Baseline(num_classes=num_classes)
    elif model_name == "stgcn-attn":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_AttnLSTM_Baseline(num_classes=num_classes, backbone=backbone)
    else:
        print(f"Unknown model: {model_name}")
        return

    model.to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"TRAINING BASELINE: {model_name} on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"[{model_name}] Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets_flat, target_lengths in pbar:
            inputs = inputs.to(DEVICE)
            targets_flat = (targets_flat + 1).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            input_lengths = torch.full((inputs.size(0),), log_probs.size(0), dtype=torch.long)
            
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
            if torch.isnan(loss) or torch.isinf(loss): continue
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"[{model_name}] Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}")
        
    # Save Final
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{model_name}_final.pth"))
    print(f"Saved {model_name} to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model type: fc-lstm, stgcn-lstm, stgcn-attn")
    args = parser.parse_args()
    
    train_model(args.model)
