import pandas as pd
import numpy as np

import torch

class AudioSegDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = torch.tensor(labels.values.astype('int').reshape(-1, 1), dtype=torch.float32)
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(next(iter(self.encodings.values())))