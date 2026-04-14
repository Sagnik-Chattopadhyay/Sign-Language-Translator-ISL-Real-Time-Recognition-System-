import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import sys
from collections import Counter
import math

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_transformer import SignTransformer
from train_sentence import SentenceDataset, collate_fn

# CONFIGURATION
SENTENCE_TENSOR_ROOT = r"e:/projects/sign language/Videos_tensors"
MAP_FILE = r"e:/projects/sign language/advanced_model/sentence_map.json"
CHECKPOINT_PATH = r"e:/projects/sign language/advanced_model/checkpoints_sentence/sentence_model_best.pth"
BATCH_SIZE = 8
MAX_FRAMES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_bleu(reference, hypothesis, n_gram=1):
    """
    Manual implementation of BLEU score calculation.
    """
    if not hypothesis or not reference: return 0
    
    # Calculate geometric mean of n-gram precisions
    precisions = []
    for n in range(1, n_gram + 1):
        ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
        hyp_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)])
        
        if not hyp_ngrams:
            precisions.append(0)
            continue
            
        matches = sum((hyp_ngrams & ref_ngrams).values())
        precisions.append(matches / sum(hyp_ngrams.values()))
    
    if min(precisions) <= 0: return 0
    
    # Geometric mean
    s = math.exp(sum(math.log(p) for p in precisions) / n_gram)
    
    # Brevity Penalty
    r = len(reference)
    c = len(hypothesis)
    bp = 1.0 if c > r else math.exp(1 - r/c)
    
    return bp * s

def ctc_greedy_decode(log_probs, vocab, blank_idx=0):
    probs = torch.exp(log_probs)
    max_probs, indices = torch.max(probs, dim=2)
    indices = indices.cpu().numpy()
    
    decoded_batch = []
    for i in range(indices.shape[0]):
        line = indices[i]
        decoded = []
        last_idx = -1
        for idx in line:
            if idx != last_idx:
                if idx != blank_idx:
                    word_idx = idx - 1
                    if 0 <= word_idx < len(vocab):
                        decoded.append(vocab[word_idx])
                last_idx = idx
        decoded_batch.append(decoded)
    return decoded_batch

def calculate_wer(reference, hypothesis):
    d = np.zeros((len(reference) + 1) * (len(hypothesis) + 1), dtype=np.uint16)
    d = d.reshape((len(reference) + 1, len(hypothesis) + 1))
    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0: d[0][j] = j
            elif j == 0: d[i][0] = i
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(reference)][len(hypothesis)] / float(len(reference))

def main():
    print(f"Evaluating Model Accuracy on {DEVICE}...")
    dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, max_frames=MAX_FRAMES, augment=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    num_classes = len(dataset.vocab) + 1
    vocab = dataset.vocab
    
    model = SignTransformer(num_classes=num_classes, phase='translation')
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE); model.eval()
    
    correct_sentences = 0
    total_samples = 0
    total_wer = 0
    
    # BLEU Scores
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    
    print("Running Inference...")
    with open("detailed_evaluation.txt", "w") as log_file:
        log_file.write("DETAILED VIDEO-BY-VIDEO EVALUATION\n")
        log_file.write("="*40 + "\n\n")
        
        with torch.no_grad():
            for inputs, targets_flat, target_lengths in tqdm(dataloader):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                log_probs = torch.log_softmax(outputs, dim=2)
                decoded_batch = ctc_greedy_decode(log_probs, vocab, blank_idx=0)
                
                curr_target_ptr = 0
                for i in range(len(target_lengths)):
                    length = target_lengths[i]
                    truth_ids = targets_flat[curr_target_ptr : curr_target_ptr + length].cpu().numpy()
                    truth_words = [vocab[tid] for tid in truth_ids]
                    curr_target_ptr += length
                    
                    pred_words = decoded_batch[i]
                    match = (pred_words == truth_words)
                    if match: correct_sentences += 1
                    wer = calculate_wer(truth_words, pred_words) if len(truth_words) > 0 else 1.0
                    total_wer += wer
                    
                    # BLEU Calculations
                    for k in range(1, 5):
                        score = calculate_bleu(truth_words, pred_words, n_gram=k)
                        bleu_scores[k].append(score)
                    
                    # Log every sample with filename
                    video_path = dataset.samples[total_samples][0]
                    video_name = os.path.basename(video_path)
                    
                    log_entry = (
                        f"Video: {video_name}\n"
                        f"Truth: {' '.join(truth_words)}\n"
                        f"Pred:  {' '.join(pred_words)}\n"
                        f"Match: {match} | WER: {wer:.2f}\n"
                        f"{'-' * 30}"
                    )
                    log_file.write(log_entry + "\n")
                    print(f"\n[{total_samples + 1}/{len(dataset)}] {log_entry}")
                    
                    total_samples += 1

    accuracy = (correct_sentences / total_samples) * 100
    avg_wer = (total_wer / total_samples) * 100
    final_bleu = {k: np.mean(v) * 100 for k, v in bleu_scores.items()}
    
    print("\n" + "="*40)
    print(f"FINAL EVALUATION RESULTS")
    print(f"Sentence Accuracy:  {accuracy:.2f}%")
    print(f"Average WER:       {avg_wer:.2f}%")
    print(f"BLEU-1:            {final_bleu[1]:.2f}")
    print(f"BLEU-2:            {final_bleu[2]:.2f}")
    print(f"BLEU-3:            {final_bleu[3]:.2f}")
    print(f"BLEU-4:            {final_bleu[4]:.2f}")
    print("="*40 + "\n")
    
    with open("accuracy_report.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "wer": avg_wer,
            "bleu": final_bleu
        }, f, indent=4)

if __name__ == "__main__":
    main()
