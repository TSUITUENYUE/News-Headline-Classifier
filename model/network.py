import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import numpy as np



class Classifier(nn.Module):
    def __init__(self,
                 combined_input,
                 combined_dim,
                 num_classes,
                 n_layers,
                 skip_in=(4,),
                 weight_norm=True):
        super(Classifier, self).__init__()
        self.num_layers = n_layers
        self.skip_in = skip_in
        self.model = LogisticRegression()
        # Combined classification layers
        dims = [combined_input] + [combined_dim for _ in range(n_layers)] + [num_classes]
        for l in range(0, self.num_layers + 1):
            if l+1 in self.skip_in:
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

            if l+1 in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers:
                x = self.activation()(x)

        # x = torch.dropout(x, p=0.2, train=self.training)
        # Output layer
        return x