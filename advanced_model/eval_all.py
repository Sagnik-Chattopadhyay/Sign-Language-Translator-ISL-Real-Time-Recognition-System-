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

from st_transformer import SignTransformer, STGCN_Encoder
from train_sentence import SentenceDataset, collate_fn
from baseline_models import (FCLSTM_Baseline, STGCN_LSTM_Baseline, 
                             STGCN_AttnLSTM_Baseline, STGCN_BiLSTM_Baseline, 
                             STGCN_GRU_Baseline, STGCN_BiGRU_Baseline,
                             STGCN_BiAttnLSTM_Baseline, CNN1D_LSTM_Baseline)

# CONFIG
SENTENCE_TENSOR_ROOT = r"e:/projects/sign language/Videos_tensors"
MAP_FILE = r"e:/projects/sign language/advanced_model/sentence_map.json"
MAX_FRAMES = 100
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "Our Method": r"e:/projects/sign language/advanced_model/checkpoints_sentence/sentence_model_best.pth",
    "Baseline: FC-LSTM": r"e:/projects/sign language/advanced_model/checkpoints_baselines/fc-lstm_final.pth",
    "ST-GCN + LSTM": r"e:/projects/sign language/advanced_model/checkpoints_baselines/stgcn-lstm_final.pth",
    "ST-GCN + Bi-LSTM": r"e:/projects/sign language/advanced_model/checkpoints_baselines/stgcn-bilstm_final.pth",
    "ST-GCN + Bi-GRU": r"e:/projects/sign language/advanced_model/checkpoints_baselines/stgcn-bigru_final.pth",
    "ST-GCN + Bi-Attn-LSTM": r"e:/projects/sign language/advanced_model/checkpoints_baselines/stgcn-biattn_final.pth",
    "CNN1D + LSTM": r"e:/projects/sign language/advanced_model/checkpoints_baselines/cnn1d-lstm_final.pth",
    "ST-GCN + GRU": r"e:/projects/sign language/advanced_model/checkpoints_baselines/stgcn-gru_final.pth",
    "ST-GCN + Attn-LSTM": r"e:/projects/sign language/advanced_model/checkpoints_baselines/stgcn-attn_final.pth"
}

def calculate_bleu(reference, hypothesis, n_gram=1):
    if not hypothesis or not reference: return 0
    precisions = []
    for n in range(1, n_gram + 1):
        ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
        hyp_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)])
        if not hyp_ngrams: precisions.append(0); continue
        matches = sum((hyp_ngrams & ref_ngrams).values())
        precisions.append(matches / sum(hyp_ngrams.values()))
    if min(precisions) <= 0: return 0
    s = math.exp(sum(math.log(p) for p in precisions) / n_gram)
    r = len(reference); c = len(hypothesis)
    bp = 1.0 if c > r else math.exp(1 - r/c)
    return bp * s

def calculate_wer(reference, hypothesis):
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.uint16)
    for i in range(len(reference) + 1): d[i][0] = i
    for j in range(len(hypothesis) + 1): d[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]: d[i][j] = d[i-1][j-1]
            else: d[i][j] = min(d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1)
    return d[len(reference)][len(hypothesis)] / float(len(reference)) if len(reference) > 0 else 1.0

def ctc_greedy_decode(log_probs, vocab, blank_idx=0):
    probs = torch.exp(log_probs)
    _, indices = torch.max(probs, dim=2)
    indices = indices.cpu().numpy()
    decoded_batch = []
    for line in indices:
        decoded = []; last_idx = -1
        for idx in line:
            if idx != last_idx and idx != blank_idx:
                word_idx = idx - 1
                if 0 <= word_idx < len(vocab): decoded.append(vocab[word_idx])
            last_idx = idx
        decoded_batch.append(decoded)
    return decoded_batch

def evaluate_model(name, path, dataset, vocab):
    num_classes = len(vocab) + 1
    if name == "Our Method":
        model = SignTransformer(num_classes=num_classes, phase='translation')
    elif name == "Baseline: FC-LSTM":
        model = FCLSTM_Baseline(num_classes=num_classes)
    elif name == "ST-GCN + LSTM":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_LSTM_Baseline(num_classes=num_classes, backbone=backbone)
    elif name == "ST-GCN + Bi-LSTM":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_BiLSTM_Baseline(num_classes=num_classes, backbone=backbone)
    elif name == "ST-GCN + Bi-GRU":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_BiGRU_Baseline(num_classes=num_classes, backbone=backbone)
    elif name == "ST-GCN + Bi-Attn-LSTM":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_BiAttnLSTM_Baseline(num_classes=num_classes, backbone=backbone)
    elif name == "CNN1D + LSTM":
        model = CNN1D_LSTM_Baseline(num_classes=num_classes)
    elif name == "ST-GCN + GRU":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_GRU_Baseline(num_classes=num_classes, backbone=backbone)
    elif name == "ST-GCN + Attn-LSTM":
        backbone = STGCN_Encoder(in_channels=3, num_class=num_classes, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)
        model = STGCN_AttnLSTM_Baseline(num_classes=num_classes, backbone=backbone)
    else: return None

    if not os.path.exists(path):
        print(f"Skipping {name}: Checkpoint not found at {path}")
        return None

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE); model.eval()
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    correct, total_wer, total_samples = 0, 0, 0
    bleus = {1:[], 2:[], 3:[], 4:[]}
    
    with torch.no_grad():
        for inputs, targets_flat, target_lengths in tqdm(loader, desc=f"Evaluating {name}"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            log_probs = torch.log_softmax(outputs, dim=2)
            decoded_batch = ctc_greedy_decode(log_probs, vocab, blank_idx=0)
            
            ptr = 0
            for i in range(len(target_lengths)):
                length = target_lengths[i]
                truth = [vocab[tid] for tid in targets_flat[ptr : ptr + length].cpu().numpy()]
                ptr += length; pred = decoded_batch[i]
                
                if pred == truth: correct += 1
                total_wer += calculate_wer(truth, pred)
                for k in range(1, 5): bleus[k].append(calculate_bleu(truth, pred, n_gram=k))
                total_samples += 1

    return {
        "Accuracy": (correct / total_samples) * 100,
        "WER": (total_wer / total_samples) * 100,
        "BLEU-1": np.mean(bleus[1]) * 100,
        "BLEU-2": np.mean(bleus[2]) * 100,
        "BLEU-3": np.mean(bleus[3]) * 100,
        "BLEU-4": np.mean(bleus[4]) * 100
    }

def main():
    dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, max_frames=MAX_FRAMES, augment=False)
    vocab = dataset.vocab
    final_results = {}

    for name, path in CHECKPOINTS.items():
        res = evaluate_model(name, path, dataset, vocab)
        if res: final_results[name] = res

    print("\n" + "="*80)
    print(f"{'Method':<20} | {'BLEU-1':<7} | {'BLEU-2':<7} | {'BLEU-3':<7} | {'BLEU-4':<7} | {'WER':<7} | {'ACC':<7}")
    print("-" * 80)
    for name, r in final_results.items():
        print(f"{name:<20} | {r['BLEU-1']:<7.2f} | {r['BLEU-2']:<7.2f} | {r['BLEU-3']:<7.2f} | {r['BLEU-4']:<7.2f} | {r['WER']:<7.2f} | {r['Accuracy']:<7.2f}")
    print("="*80 + "\n")
    
    with open("final_comparison.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("Results saved to final_comparison.json")

if __name__ == "__main__":
    main()
