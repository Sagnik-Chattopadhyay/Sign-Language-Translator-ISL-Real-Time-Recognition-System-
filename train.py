import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SignLanguageDataset
from stgcn import Model
from tqdm import tqdm

# CONFIGURATION
DATA_ROOT = r"e:/projects/sign language/Videos_tensors"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 4
LEARNING_RATE = 0.001 # Adam default
NUM_EPOCHS = 100
NUM_CLASSES = 101 # Based on dataset scan (was 101 classes)
MAX_FRAMES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Load Dataset
    print("Loading Dataset...")
    dataset = SignLanguageDataset(DATA_ROOT, max_frames=MAX_FRAMES, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 2. Model Initialization
    # In_Channels=3 (x,y,z), Num_Classes=101+1 (CTC needs blank)
    # CTC blank is usually class 0 or class N.
    # PyTorch CTCLoss defaults blank=0. So valid classes should be 1..N.
    # We shift our dataset labels by +1.
    ctc_num_classes = NUM_CLASSES + 1
    
    graph_args = {'strategy': 'spatial'}
    model = Model(in_channels=3, num_class=ctc_num_classes, graph_args=graph_args, edge_importance_weighting=True)
    model.to(DEVICE)
    
    # 3. Loss and Optimizer
    # blank=0 is default for CTCLoss.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # Reduce LR every 20 epochs
    
    print(f"Starting Training on {DEVICE}...")

    # 4. Resume fom Checkpoint
    start_epoch = 0
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("stgcn_epoch_") and f.endswith(".pth")]
    if checkpoints:
        # Extract epoch numbers
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_epoch = int(latest_checkpoint.split('_')[2].split('.')[0])
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
        
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        start_epoch = latest_epoch
    else:
        print("No checkpoint found. Starting from scratch.")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, labels in pbar:
            # inputs: (N, 3, T, 119, 1)
            # labels: (N) - integer class indices
            
            inputs = inputs.to(DEVICE)
            # Shift labels by +1 because 0 is blank for CTC
            labels = labels.to(DEVICE) + 1 
            
            optimizer.zero_grad()
            
            # Forward
            # Outputs: (N, T_out, Num_Classes) - LogSoftmaxed
            outputs = model(inputs)
            
            # Prepare for CTC inputs
            # 1. Log_Probs: (T, N, Num_Classes) requires Time-Major
            log_probs = outputs.permute(1, 0, 2)
            
            # 2. Input Lengths (T for each item in batch)
            # Our model output T is fixed (approx T_in / 4)
            T_out = outputs.size(1)
            input_lengths = torch.full((inputs.size(0),), T_out, dtype=torch.long)
            
            # 3. Target Lengths (Length of each target sequence)
            # For classification, len=1
            target_lengths = torch.full((inputs.size(0),), 1, dtype=torch.long)
            
            # Calculate Loss
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"stgcn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

    print("Training Completed.")

if __name__ == "__main__":
    main()
