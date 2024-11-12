import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


class CLSDataset(Dataset):
    def __init__(self, csv_file,transform=None):
        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(csv_file)
        self.seqtokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tfidf_vectorizer = TfidfVectorizer()
        self.transform = transform

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
            freq_input = self.tfidf_vectorizer.fit_transform([headline]).toarray()[0]
            seq_input = self.seqtokenizer(headline, padding='max_length', truncation=True, return_tensors='pt')
            pos_input = headline
        else:
            freq_input = headline
            seq_input = headline
            pos_input = headline
        return (freq_input,seq_input,pos_input), agency