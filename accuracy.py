import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SignLanguageDataset
from stgcn import Model
from tqdm import tqdm
import os

# CONFIGURATION
DATA_ROOT = r"e:/projects/sign language/Videos_tensors"
CHECKPOINT_PATH = r"e:/projects/sign language/checkpoints/stgcn_epoch_100.pth"
BATCH_SIZE = 8
NUM_CLASSES = 101 # Dataset classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 100

def ctc_greedy_decode(output, blank_label=0):
    """
    Decodes CTC output (T, N, C) -> List of lists of labels
    output: LogSoftmax (T, N, C) or Softmax
    """
    # output (T, N, C) -> argmax -> (T, N)
    output = output.detach().cpu()
    predictions = torch.argmax(output, dim=2) # (T, N)
    
    decoded_batch = []
    
    for i in range(predictions.size(1)): # Iterate over batch
        raw_pred = predictions[:, i].numpy()
        decoded = []
        previous = blank_label
        
        for p in raw_pred:
            if p != blank_label and p != previous:
                decoded.append(p)
            previous = p
            
        decoded_batch.append(decoded)
        
    return decoded_batch

def main():
    print(f"Checking accuracy on {DEVICE}...")
    
    # 1. Load Dataset
    # No augmentation for evaluation
    dataset = SignLanguageDataset(DATA_ROOT, max_frames=MAX_FRAMES, augment=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Dataset: {len(dataset)} samples.")
    
    # 2. Load Model
    ctc_num_classes = NUM_CLASSES + 1
    graph_args = {'strategy': 'spatial'}
    model = Model(in_channels=3, num_class=ctc_num_classes, graph_args=graph_args, edge_importance_weighting=True)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    model.to(DEVICE)
    model.eval()
    
    correct_count = 0
    total_count = 0
    
    print("Evaluating...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            # Labels in dataset are 0-100. In Model they are 1-101.
            # Convert targets to list of lists for comparison
            target_labels = (labels + 1).numpy().tolist()
            
            # Forward
            outputs = model(inputs) # (N, T, C)
            
            # Permute for Decoding logic expectation (T, N, C) if reusing logic, 
            # but my greedy decode above expects (T, N, C). stgcn output is (N, T, C).
            # Let's permute inputs to decoding function
            outputs = outputs.permute(1, 0, 2) # (T, N, C)
            
            decoded_batch = ctc_greedy_decode(outputs, blank_label=0)
            
            for i in range(len(target_labels)):
                pred = decoded_batch[i]
                truth = [target_labels[i]] # Since we trained with length 1 sequence
                
                # Check for exact match
                if pred == truth:
                    correct_count += 1
                
                # Debug Print for first few
                if total_count < 5:
                     print(f"Pred: {pred} | Truth: {truth} | {'CORRECT' if pred==truth else 'WRONG'}")

                total_count += 1
                
    accuracy = (correct_count / total_count) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

if __name__ == "__main__":
    main()
