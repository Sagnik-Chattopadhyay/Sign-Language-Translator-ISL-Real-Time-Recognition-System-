import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graph import Graph

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # Spatial Graph Convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # Temporal Convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # Residual Connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res), A

class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting=True, **kwargs):
        super().__init__()

        # Load Graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Build Networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # Architecture: Variable number of blocks (1 to 10 supported)
        # Standard: 10 Blocks with 64 -> 64 -> 128 -> 128 -> 256 -> 256
        # If num_layers=5, we take the first 5 with appropriate strides.
        num_layers = kwargs.get('num_layers', 10)
        
        full_architecture = [
            (in_channels, 64, 1, False),
            (64, 64, 1, True),
            (64, 64, 1, True),
            (64, 64, 1, True),
            (64, 128, 2, True),
            (128, 128, 1, True),
            (128, 128, 1, True),
            (128, 256, 2, True),
            (256, 256, 1, True),
            (256, 256, 1, True),
        ]
        
        self.st_gcn_networks = nn.ModuleList()
        current_out_channels = 64
        for i in range(min(num_layers, len(full_architecture))):
            in_c, out_c, s, res = full_architecture[i]
            self.st_gcn_networks.append(STGCN_Block(in_c, out_c, kernel_size, stride=s, residual=res))
            current_out_channels = out_c

        # Edge Importance Weights (Learnable Mask for Adjacency)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Classification Head for CTC (Frame-wise)
        # Use current_out_channels instead of hardcoded 256
        self.fc = nn.Conv1d(current_out_channels, num_class, kernel_size=1)

    def forward(self, x):
        # Input x: (N, C, T, V, M)
        
        # 1. Data Normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous() # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous() # N, M, C, T, V
        x = x.view(N * M, C, T, V)

        # 2. ST-GCN Blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # 3. Global Pooling mechanism
        # For CSLR, we pool over Vertices (Space) and Person (M), 
        # but KEEP Time (T) for CTC alignment.
        
        # x is (N*M, 256, T_out, V)
        
        # Pool V dimension
        # x is (N*M, last_channels, T_out, V)
        x = F.avg_pool2d(x, (1, V)) 
        x = x.view(N, M, current_out_channels, -1) # (N, M, C, T_out)
        
        # Pool M dimension (Mean)
        x = x.mean(dim=1) # (N, C, T_out)

        # 4. Classifier (Frame-wise)
        # x: (N, 256, T_out)
        x = self.fc(x) # (N, Num_Class, T_out)
        
        # 5. Prepare for CTC Loss
        # CTC expects (T, N, Num_Class) usually, but we can return (N, Num_Class, T) 
        # or (N, T, Num_Class).
        x = x.permute(0, 2, 1) # (N, T_out, Num_Class)
        
        # Log Softmax is often required by CTCLoss
        x = F.log_softmax(x, dim=2)

        return x

if __name__ == "__main__":
    # Simple Verification
    batch = 2
    channels = 3
    frames = 100
    vertices = 119
    people = 1
    num_class = 10 # Example vocabulary size
    
    # Random Input
    x = torch.randn(batch, channels, frames, vertices, people)
    
    graph_args = {'strategy': 'spatial'}
    model = Model(channels, num_class, graph_args, edge_importance_weighting=True)
    
    output = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape (N, T, C): {output.shape}")
    
    # Expected T reduction: Stride 2 at layer 4 and 7 -> /4
    # 100 / 4 = 25
    expected_frames = frames // 4
    assert output.shape == (batch, expected_frames, num_class)
    print("Verification Successful!")
