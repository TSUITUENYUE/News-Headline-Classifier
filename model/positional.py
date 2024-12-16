import torch
import torch.nn as nn
import numpy as np
from model.siren import SIREN

class PosNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 n_layers,
                 skip_in=(4,),
                 weight_norm=True):
        super(PosNetwork, self).__init__()
        dims = [input_dim] + [hidden_dim for _ in range(n_layers)] + [output_dim]
        self.num_layers = n_layers
        self.skip_in = skip_in
        self.siren_layers = nn.ModuleList()
        for l in range(0, self.num_layers + 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] + dims[0]
                dims[l + 1] = out_dim
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.activation = nn.ReLU

    def forward(self, inputs):
        x = inputs
        for l in range(0, self.num_layers + 1):
            lin = getattr(self, "lin" + str(l))
            if l + 1 in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers:
                x = self.activation()(x)
        x = x.mean(dim=1)  # Pooling to match Shape: (32, 128)
        return x