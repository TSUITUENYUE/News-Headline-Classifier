import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class SeqNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(SeqNetwork, self).__init__()

        # BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Optionally freeze BERT layers
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # Sequential model for BERT embeddings
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.bert_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_sequence_output = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_length, feature_size)

        # LSTM over BERT embeddings
        lstm_out, (h_n, c_n) = self.lstm(bert_sequence_output)
        bert_feature = h_n[-1]  # Use the last hidden state
        bert_feature = nn.ReLU()(self.bert_fc(bert_feature))  # Shape: (batch_size, feature_size)


        return bert_feature