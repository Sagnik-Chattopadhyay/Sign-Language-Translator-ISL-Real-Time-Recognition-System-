import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLSTM_Baseline(nn.Module):
    """
    Simple Baseline: Flattens the coordinates and feeds them into an LSTM.
    Input: (N, C, T, V, M) -> (N, T, C*V*M)
    """
    def __init__(self, num_classes, input_dim=3*119*1, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(N, T, -1) # (N, T, C*V*M)
        
        out, _ = self.lstm(x) # (N, T, hidden_dim)
        out = self.fc(out) # (N, T, num_classes)
        return out

class STGCN_LSTM_Baseline(nn.Module):
    """
    ST-GCN backbone + LSTM decoder (No Transformer/Attention).
    Identical to our model but uses LSTM instead of Transformer Encoder.
    """
    def __init__(self, num_classes, backbone, d_model=256, num_layers=2):
        super().__init__()
        self.encoder = backbone # Expects STGCN_Encoder from st_transformer.py
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # 1. Feature Extraction: (N, 256, T_out)
        features = self.encoder(x)
        
        # 2. LSTM Decoding
        features = features.permute(0, 2, 1) # (N, T_out, 256)
        out, _ = self.lstm(features) # (N, T_out, 256)
        
        # 3. Classify
        out = self.classifier(out) # (N, T_out, num_classes)
        return out

class STGCN_BiLSTM_Baseline(nn.Module):
    """
    ST-GCN backbone + Bidirectional LSTM decoder.
    """
    def __init__(self, num_classes, backbone, d_model=256, num_layers=2):
        super().__init__()
        self.encoder = backbone
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(d_model * 2, num_classes)
        
    def forward(self, x):
        features = self.encoder(x) # (N, 256, T_out)
        features = features.permute(0, 2, 1) # (N, T_out, 256)
        out, _ = self.lstm(features) # (N, T_out, 512)
        out = self.classifier(out) # (N, T_out, num_classes)
        return out

class STGCN_GRU_Baseline(nn.Module):
    """
    ST-GCN backbone + GRU decoder.
    """
    def __init__(self, num_classes, backbone, d_model=256, num_layers=2):
        super().__init__()
        self.encoder = backbone
        self.gru = nn.GRU(d_model, d_model, num_layers, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        features = self.encoder(x) # (N, 256, T_out)
        features = features.permute(0, 2, 1) # (N, T_out, 256)
        out, _ = self.gru(features) # (N, T_out, 256)
        out = self.classifier(out) # (N, T_out, num_classes)
        return out

class STGCN_BiGRU_Baseline(nn.Module):
    """
    ST-GCN backbone + Bidirectional GRU decoder.
    """
    def __init__(self, num_classes, backbone, d_model=256, num_layers=2):
        super().__init__()
        self.encoder = backbone
        self.gru = nn.GRU(d_model, d_model, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(d_model * 2, num_classes)
        
    def forward(self, x):
        features = self.encoder(x) # (N, 256, T_out)
        features = features.permute(0, 2, 1) # (N, T_out, 256)
        out, _ = self.gru(features) # (N, T_out, 512)
        out = self.classifier(out) # (N, T_out, num_classes)
        return out

class CNN1D_LSTM_Baseline(nn.Module):
    """
    Non-graph baseline: 1D CNN for spatial features + LSTM for temporal.
    Input: (N, C, T, V, M) -> View as (N, T, C*V*M) -> Conv1d on (N, C*V*M, T)
    """
    def __init__(self, num_classes, input_dim=3*119*1, d_model=256, num_layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(N, T, -1).permute(0, 2, 1) # (N, C*V*M, T)
        
        feat = self.conv(x) # (N, 256, T)
        feat = feat.permute(0, 2, 1) # (N, T, 256)
        
        out, _ = self.lstm(feat)
        out = self.classifier(out)
        return out

class STGCN_AttnLSTM_Baseline(nn.Module):
    """
    ST-GCN backbone + LSTM with simple Attention decoder.
    """
    def __init__(self, num_classes, backbone, d_model=256, num_layers=2):
        super().__init__()
        self.encoder = backbone
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        
        # Simple attention projection
        self.attn_proj = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        features = self.encoder(x) # (N, 256, T_out)
        features = features.permute(0, 2, 1) # (N, T, 256)
        
        lstm_out, _ = self.lstm(features) # (N, T, 256)
        
        # Simple Dot-Product Style Attention
        # (N, T, 256) @ (N, 256, T) -> (N, T, T) weight matrix
        weights = torch.bmm(lstm_out, features.transpose(1, 2))
        weights = F.softmax(weights, dim=-1)
        
        context = torch.bmm(weights, features) # (N, T, 256)
        
        out = self.classifier(context + lstm_out) # Residual-like connection
        return out

class STGCN_BiAttnLSTM_Baseline(nn.Module):
    """
    ST-GCN backbone + Bidirectional LSTM with Attention.
    """
    def __init__(self, num_classes, backbone, d_model=256, num_layers=2):
        super().__init__()
        self.encoder = backbone
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(d_model * 2, d_model * 2)
        self.classifier = nn.Linear(d_model * 2, num_classes)
        
    def forward(self, x):
        features = self.encoder(x) # (N, 256, T_out)
        features = features.permute(0, 2, 1) # (N, T, 256)
        
        lstm_out, _ = self.lstm(features) # (N, T, 512)
        
        # Expand features to 512 to match Bi-LSTM for attention
        feat_expanded = torch.cat([features, features], dim=-1) # (N, T, 512)
        
        weights = torch.bmm(lstm_out, feat_expanded.transpose(1, 2))
        weights = F.softmax(weights, dim=-1)
        
        context = torch.bmm(weights, feat_expanded) # (N, T, 512)
        
        out = self.classifier(context + lstm_out)
        return out
