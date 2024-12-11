import torch
import torch.nn as nn
import numpy as np



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

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)  # Learnable attention weights

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        scores = self.attention_weights(x)  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        attention_probs = torch.softmax(scores, dim=1)  # Normalize across seq_len
        x_pooled = torch.sum(x * attention_probs.unsqueeze(-1), dim=1)  # Weighted sum
        return x_pooled


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
            if l == 0:
                self.siren_layers.append(SIREN(dims[l], dims[l + 1], is_first=True))
            else:
                self.siren_layers.append(SIREN(dims[l], dims[l + 1]))

    def forward(self, inputs):
        x = inputs
        for l in range(0, self.num_layers + 1):
            siren_layer = self.siren_layers[l]
            x = siren_layer(x)
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = torch.dropout(x, p=0.2, train=self.training)

        attention_pooling = AttentionPooling(input_dim=x.shape[1])  # Initialize the module
        x = attention_pooling(x) # Pooling to match Shape: (32, 128)

        return x