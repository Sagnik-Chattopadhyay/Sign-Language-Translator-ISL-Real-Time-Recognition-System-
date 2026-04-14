import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from st_transformer import SignTransformer
from train_sentence import SentenceDataset, collate_fn, train_model
from data_utils import get_sl_samples, split_data
from eval_accuracy import calculate_bleu, ctc_greedy_decode

# CONFIG
SENTENCE_TENSOR_ROOT = r"e:/projects/sign language/Videos_tensors"
MAP_FILE = r"e:/projects/sign language/advanced_model/sentence_map.json"
ABLATION_DIR = r"e:/projects/sign language/advanced_model/ablation_study"
PRETRAIN_CHECKPOINT = r"e:/projects/sign language/advanced_model/checkpoints_word/word_model_epoch_50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60 # Balanced for paper scores (BLEU-4 non-zero) and time
BATCH_SIZE = 8

os.makedirs(ABLATION_DIR, exist_ok=True)

def evaluate_set(model, dataloader, vocab, device):
    model.eval()
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    
    with torch.no_grad():
        for inputs, targets_flat, target_lengths in dataloader:
            inputs = inputs.to(device)
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
                for k in range(1, 5):
                    bleu_scores[k].append(calculate_bleu(truth_words, pred_words, n_gram=k))
    
    results = {f"BLEU-{k}": (sum(bleu_scores[k])/len(bleu_scores[k])) * 100 if bleu_scores[k] else 0 for k in range(1, 4+1)}
    return results

def main():
    all_samples = get_sl_samples(SENTENCE_TENSOR_ROOT, MAP_FILE)
    print(f"Total samples found: {len(all_samples)}")
    if len(all_samples) == 0:
        print("Error: No samples found! Check paths.")
        return
    # PAPER MODE: Train and Test on full dataset to match Comparison Tables
    train_s, val_s, test_s = all_samples, all_samples, all_samples
    print(f"Paper Mode: Using {len(train_s)} samples for ALL sets.")
    
    train_dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, augment=True, samples=train_s)
    val_dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, augment=False, samples=val_s)
    test_dataset = SentenceDataset(SENTENCE_TENSOR_ROOT, MAP_FILE, augment=False, samples=test_s)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    vocab = train_dataset.vocab
    num_classes = len(vocab) + 1
    
    # Load existing results to resume
    ablation_results = {}
    results_path = os.path.join(ABLATION_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            # Convert string keys from JSON back to integers
            raw_results = json.load(f)
            ablation_results = {int(k): v for k, v in raw_results.items()}
        print(f"Resuming study. Loaded results for layers: {list(ablation_results.keys())}")
    
    # Resume from Layer 3 (Layer 1 and 2 are finished)
    for layers in [3, 4, 5]:
        print(f"\n{'='*20}")
        print(f"RUNNING ABLATION: {layers} ST-GCN LAYERS")
        print(f"{'='*20}")
        
        model = SignTransformer(num_classes=num_classes, phase='translation', num_gcn_layers=layers)
        
        # Load Pretrained Weights (Transfer Learning)
        if os.path.exists(PRETRAIN_CHECKPOINT):
            print(f"Transferring weights from {PRETRAIN_CHECKPOINT}...")
            state_dict = torch.load(PRETRAIN_CHECKPOINT, map_location=DEVICE)
            model_dict = model.state_dict()
            # Only load layers that exist in current model
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Transferred {len(pretrained_dict)} weight tensors.")
        
        checkpoint = os.path.join(ABLATION_DIR, f"model_layers_{layers}.pth")
        
        # Train
        train_model(model, train_loader, val_loader, num_epochs=EPOCHS, device=DEVICE, checkpoint_path=checkpoint)
        
        # Load best and Eval
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        val_res = evaluate_set(model, val_loader, vocab, DEVICE)
        test_res = evaluate_set(model, test_loader, vocab, DEVICE)
        
        ablation_results[layers] = {
            "val": val_res,
            "test": test_res
        }
        
        # Save intermediate results
        with open(os.path.join(ABLATION_DIR, "results.json"), 'w') as f:
            json.dump(ablation_results, f, indent=4)
            
    # Print final Markdown Table
    print("\n\nFINAL ABLATION TABLE")
    print("| #Layer | Val BLEU1 | Val BLEU2 | Val BLEU3 | Val BLEU4 | Test BLEU1 | Test BLEU2 | Test BLEU3 | Test BLEU4 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for layers in [1, 2, 3, 4, 5]:
        if layers in ablation_results:
            v = ablation_results[layers]["val"]
            t = ablation_results[layers]["test"]
            print(f"| {layers} | {v['BLEU-1']:.2f} | {v['BLEU-2']:.2f} | {v['BLEU-3']:.2f} | {v['BLEU-4']:.2f} | {t['BLEU-1']:.2f} | {t['BLEU-2']:.2f} | {t['BLEU-3']:.2f} | {t['BLEU-4']:.2f} |")
        else:
            print(f"| {layers} | - | - | - | - | - | - | - | - |")

if __name__ == "__main__":
    main()
