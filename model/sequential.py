import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

class SeqNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 n_layers,
                 skip_in=(4,),
                 weight_norm=True,
                 freeze = False):
        super(SeqNetwork, self).__init__()

        self.freeze = freeze
        self.skip_in = skip_in
        self.num_layers = n_layers
        self.weight_norm = weight_norm
        # BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if self.freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Sequential model for BERT embeddings
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        dims = [input_dim] + [hidden_dim for _ in range(n_layers)] + [output_dim]
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.activation = nn.ReLU

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_sequence_output = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_length, feature_size)

        # LSTM over BERT embeddings
        lstm_out, (h_n, c_n) = self.lstm(bert_sequence_output)
        inputs = h_n[-1]  # Use the last hidden state
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        bert_feature = x # Shape: (batch_size, feature_size)
        return bert_feature