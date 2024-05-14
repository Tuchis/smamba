import requests
import os
from utils.tokenizer import Tokenizer
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, data: str, tokenizer: Tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized = self.tokenizer.tokenize(data)

    def __len__(self):
        return len(self.tokenized)-self.max_length

    def __getitem__(self, idx):
        text = self.tokenized[idx:idx+self.max_length]
        text = torch.tensor(text, dtype=torch.int)
        label = self.tokenized[idx+self.max_length]
        label = torch.tensor(label, dtype=torch.long)
        return text, label

def get_text(url: str):
    if os.path.exists(url):
        with open(url, "r") as f:
            return f.read()
    return requests.get(url).text

def get_dataset(data: str, tokenizer: Tokenizer, max_length: int = 512):
    return TextDataset(data, tokenizer, max_length)

if __name__ == "__main__":
    tokenizer = Tokenizer("tokenizer.json")
    data = get_text("https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt")
    dataset = get_dataset(data, tokenizer, max_length=512)
    print(dataset[0])
