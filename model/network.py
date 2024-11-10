import torch
import torch.nn as nn
import torch.nn.functional as F
from model.positional import PosNetwork
from model.sequential import SeqNetwork
from model.frequential import FreqNetwork

class Classifier(nn.Module):
    def __init__(self,
                 tfidf_input_dim,
                 tfidf_hidden_dim,
                 tfidf_output_dim,
                 seq_input_dim,
                 seq_hidden_dim,
                 seq_output_dim,
                 combined_dim,
                 num_classes):
        super(Classifier, self).__init__()

        # Frequential Network
        self.freq = FreqNetwork(tfidf_input_dim,tfidf_hidden_dim,tfidf_output_dim)

        # Sequential Network
        self.seq = SeqNetwork(seq_input_dim,seq_hidden_dim,seq_output_dim)

        # Positional Network
        # self.pos = PosNetwork()
        # Combined classification layers
        combined_input = tfidf_output_dim + seq_output_dim
        self.combined_fc1 = nn.Linear(combined_input, combined_dim)
        self.output = nn.Linear(combined_dim, num_classes)

    def forward(self, input_ids, attention_mask, tfidf_features):

        seq_feature = self.seq(input_ids, attention_mask)
        freq_feature = self.freq(tfidf_features)
        # Concat features
        combined_feature = torch.cat((seq_feature, freq_feature), dim=1)  # Shape: (batch_size, 128)
        combined_feature = nn.ReLU((self.combined_fc1(combined_feature)))

        # Output layer
        prob = F.sigmoid(self.output(combined_feature))

        return prob