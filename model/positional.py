import torch
import torch.nn as nn
import numpy as np

def positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return torch.tensor(pos_enc, dtype=torch.float)

class SIREN(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super(SIREN, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0,
                                            np.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

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
        for l in range(0, self.num_layers - 1):
            if l == 0:
                self.siren_layers.append(SIREN(dims[l], dims[l + 1], is_first=True))
            else:
                self.siren_layers.append(SIREN(dims[l], dims[l + 1]))

    def forward(self, inputs):
        inputs = positional_encoding(inputs.size(1), inputs.size(2)).to(inputs.device)
        x = inputs
        for l in range(0, self.num_layers - 1):
            siren_layer = self.siren_layers[l]
            x = siren_layer(x)
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
        return x