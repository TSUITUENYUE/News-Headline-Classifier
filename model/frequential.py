import torch
import torch.nn as nn


class FreqNetwork(nn.Module):
    def __init__(self, tfidf_input_dim,tfidf_hidden_dim,tfidf_output_dim):
        super(FreqNetwork, self).__init__()

        self.tfidf_fc1 = nn.Linear(tfidf_input_dim, tfidf_hidden_dim)
        self.tfidf_fc2 = nn.Linear(tfidf_hidden_dim, tfidf_output_dim)


    def forward(self, x):

        h = nn.ReLU((self.tfidf_fc1(x)))
        h = nn.ReLU((self.tfidf_fc2(h))) # Shape: (batch_size, 64)


        return h