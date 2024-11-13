import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return torch.tensor(pos_enc, dtype=torch.float)

class CLSDataset(Dataset):
    def __init__(self, data_dir,transform=True):
        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(data_dir)
        self.seqtokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        self.transform = transform
        self.tfidf_vectorizer.fit(self.data['title'])

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row at the specified index
        row = self.data.iloc[idx]
        # Extract features and target
        headline = row['title']  # First column as x
        agency = row['news']  # Second column as y
        # Return x and y as a dictionary or tuple
        if self.transform:
            freq_input = self.tfidf_vectorizer.transform([headline]).toarray()[0] #dim: 8621
            seq_input = self.seqtokenizer(headline, padding='max_length', truncation=True, return_tensors='pt') #dim: 512
            seq_input = torch.vstack((seq_input['input_ids'], seq_input['attention_mask']))
            pos_input = positional_encoding(64, 256)
        else:
            freq_input = headline
            seq_input = headline
            pos_input = headline
        return (torch.tensor(freq_input, dtype=torch.float32),
                torch.tensor(seq_input,dtype=torch.int),
                torch.tensor(pos_input,dtype=torch.float32)), torch.tensor(1.0 if agency == "fox" else 0.0, dtype=torch.float32)