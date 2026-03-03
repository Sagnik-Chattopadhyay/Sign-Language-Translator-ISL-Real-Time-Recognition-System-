import os
import json

# CONFIGURATION
SENTENCE_ROOT = r"e:/projects/sign language/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level"
WORD_ROOT = r"e:/projects/sign language/advanced_model/data_words"
OUTPUT_FILE = r"e:/projects/sign language/advanced_model/sentence_map.json"

def main():
    # 1. Load Vocabulary
    vocab = sorted([d for d in os.listdir(WORD_ROOT) if os.path.isdir(os.path.join(WORD_ROOT, d))])
    vocab_map = {word.lower(): js_idx for js_idx, word in enumerate(vocab)}
    
    print(f"Loaded Vocabulary: {len(vocab)} words.")
    
    # 2. Load Sentences
    sentences = sorted([d for d in os.listdir(SENTENCE_ROOT) if os.path.isdir(os.path.join(SENTENCE_ROOT, d))])
    print(f"Loaded Sentences: {len(sentences)} classes.")
    
    mapping = {}
    total_words = 0
    unknown_words = 0
    
    # Expand Vocab list
    # We want to keep original indices for consistency with Pretraining
    # So we append new words at the end.
    expanded_vocab = list(vocab)
    vocab_map = {word.lower(): i for i, word in enumerate(expanded_vocab)}
    
    for sentence in sentences:
        tokens = sentence.lower().split()
        
        token_ids = []
        for token in tokens:
            total_words += 1
            clean_token = token.strip('.,?!')
            
            # Check original vocab first
            if clean_token in vocab_map:
                token_ids.append(vocab_map[clean_token])
            else:
                # Add to vocab!
                print(f"Adding new word: '{clean_token}'")
                expanded_vocab.append(clean_token.upper())
                new_idx = len(expanded_vocab) - 1
                vocab_map[clean_token] = new_idx
                token_ids.append(new_idx)
                unknown_words += 1
                    
        if token_ids:
            mapping[sentence] = token_ids
            
    # 3. Save Mapping
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({'vocab': expanded_vocab, 'mapping': mapping}, f, indent=4)
        
    print(f"\nExample Mapping for 'comb your hair': {mapping.get('comb your hair')}")
        
    print(f"\nMapping Complete.")
    print(f"Total Words Processed: {total_words}")
    print(f"Unknown Words: {unknown_words} ({unknown_words/total_words:.2%})")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
