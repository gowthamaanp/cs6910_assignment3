import torch
from torch.utils.data import TensorDataset, Dataset
import pandas as pd
import numpy as np

from .charset import CharSet

class TransliterationDataset(Dataset):
    def __init__(self, src_lang, trg_lang):
        self.read_csv_data(src_lang, trg_lang)
       
    def __len__(self):
        return len(self.src_words)

    def __getitem__(self, idx):
        return self.src_words[idx], self.trg_words[idx]

    def read_csv_data(self, src_lang, trg_lang):
        corpus_train = pd.read_csv(f"././data/aksharantar_sampled/{trg_lang}/{trg_lang}_train.csv",  header=None)
        corpus_test = pd.read_csv(f"././data/aksharantar_sampled/{trg_lang}/{trg_lang}_test.csv",  header=None)
        corpus_valid = pd.read_csv(f"././data/aksharantar_sampled/{trg_lang}/{trg_lang}_valid.csv",  header=None)
        
        self.src_charset = CharSet(name=src_lang)
        self.trg_charset = CharSet(name=trg_lang)
        
        self.src_words, self.trg_words = corpus_train.iloc[:, 0], corpus_train.iloc[:, 1]
        self.train_dataset = self.build_dataset(corpus_train, src_charset, trg_charset)
        self.test_dataset = self.build_dataset(corpus_test, src_charset, trg_charset)
        self.valid_dataset = self.build_dataset(corpus_valid, src_charset, trg_charset)
       
        
    def build_dataset(self, data, src_charset, trg_charset):
        # Maximum Sequence Length
        max_seq_length = 35
        
        # Create lookup tables for characters
        src_char2idx = src_charset.char2index
        trg_char2idx = trg_charset.char2index
        
        # Pad and encode input sequences
        input_tensor = torch.zeros((len(data[0]), max_seq_length), dtype=torch.long)
        output_tensor = torch.zeros((len(data[1]), max_seq_length), dtype=torch.long)
        
        for i in range(len(data)):
            in_seq = data[0][i].ljust(max_seq_length, '#')
            out_seq = data[1][i].ljust(max_seq_length, '#')
            for j, char in enumerate(in_seq):
                input_tensor[i, j] = src_char2idx.get(char)
            for j, char in enumerate(out_seq):
                output_tensor[i, j] = trg_char2idx.get(char)   
                
        return TensorDataset(input_tensor, output_tensor)
        

    def get_random_sample(self):
        return self.__getitem__(np.random.randint(len(self.src_words)))