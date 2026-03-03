import torch
import torch.nn as nn
import torch.nn.functional as F
from stgcn import Model as STGCN_Model

class STGCN_Encoder(nn.Module):
    """
    Wraps the ST-GCN model to act as a pure feature extractor.
    Removes the final classification head.
    """
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting):
        super().__init__()
        # Initialize original model
        self.model = STGCN_Model(in_channels, num_class, graph_args, edge_importance_weighting)
        
        # Remove or bypass the final classification layer (self.model.fc)
        # We want the features before that: (N, 256, T, V) -> Pooled -> (N, 256, T)
        
    def forward(self, x):
        # x: (N, C, T, V, M)
        
        # 1. Reuse ST-GCN layers
        # Copy-paste logic from stgcn.py forward() but stop before fc
        
        # Data Normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.model.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Layers
        for gcn, importance in zip(self.model.st_gcn_networks, self.model.edge_importance):
            x, _ = gcn(x, self.model.A * importance)

        # Global Pooling (V, M) -> Keep T
        # x is (N*M, 256, T_out, V)
        x = F.avg_pool2d(x, (1, V)) 
        x = x.view(N, M, 256, -1) # (N, M, C, T_out)
        x = x.mean(dim=1) # (N, C, T_out)
        
        return x # (N, 256, T_out)

class SignTransformer(nn.Module):
    def __init__(self, num_classes, phase='pretrain', d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.phase = phase
        
        # 1. Encoder (Pre-trained Backbone)
        graph_args = {'strategy': 'spatial'}
        self.encoder = STGCN_Encoder(in_channels=3, num_class=num_classes, 
                                     graph_args=graph_args, edge_importance_weighting=True)
        
        # 2. Phase 1: Word Classifier
        # Used for Pre-training on isolated words
        self.classifier = nn.Linear(d_model, num_classes)
        
        # 3. Phase 2: Transformer Decoder (Translation)
        # Used for Sequence-to-Sequence on sentences
        # Target Vocab might be different from Pre-train Classes? 
        # For now, assume same vocabulary.
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.translation_head = nn.Linear(d_model, num_classes) # Output words
        
        # Positional Encoding for Decoder?
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, x, target_sequence=None):
        # x: (N, C, T, V, M)
        
        # Extract Features: (N, 256, T_out)
        features = self.encoder(x)
        features = features.permute(2, 0, 1) # (T_out, N, 256) for Transformer
        
        if self.phase == 'pretrain':
            # Global Temporal Pooling (Mean over Time)
            # features: (T, N, D) -> (N, D)
            pooled = features.mean(dim=0)
            output = self.classifier(pooled) # (N, Num_Classes)
            return output
            
        elif self.phase == 'translation':
            # Transformer Decoding
            # target_sequence: (T_tgt, N) embedding indices
            # Need embeddings for target...
            # This part requires more setup (Embbedings, Masking).
            # For iteration 1, we focus on Pretrain.
            pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Simplified: Just learnable or fixed
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]

if __name__ == "__main__":
    # Test
    model = SignTransformer(num_classes=114, phase='pretrain')
    x = torch.randn(2, 3, 50, 119, 1)
    out = model(x)
    print(f"Pretrain Output: {out.shape}") # Should be (2, 114)
