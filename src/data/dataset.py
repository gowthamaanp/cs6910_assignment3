import torch
from torch.utils.data import TensorDataset, Dataset
import pandas as pd
import numpy as np
from .charset import CharSet, MAX_SEQUENCE_LENGTH

class TransliterationDataset(Dataset):
    def __init__(self, src_lang, trg_lang):
        self.read_csv_file(src_lang, trg_lang)  # Read CSV file and build datasets

    def __len__(self):
        return len(self.src_words)  # Return the length of the source word list

    def __getitem__(self, idx):
        return self.src_words[idx], self.trg_words[idx]  # Return a pair of source and target words at the given index

    def read_csv_file(self, src_lang, trg_lang):
        # Read CSV files for train, test, and validation sets
        corpus_train = pd.read_csv(f"././data/aksharantar_sampled/{trg_lang}/{trg_lang}_train.csv", header=None)
        corpus_test = pd.read_csv(f"././data/aksharantar_sampled/{trg_lang}/{trg_lang}_test.csv", header=None)
        corpus_valid = pd.read_csv(f"././data/aksharantar_sampled/{trg_lang}/{trg_lang}_valid.csv", header=None)

        self.src_charset = CharSet(language=src_lang)  # Initialize source language character set
        self.trg_charset = CharSet(language=trg_lang)  # Initialize target language character set

        self.src_words, self.trg_words = corpus_train.iloc[:, 0], corpus_train.iloc[:, 1]  # Extract source and target words from the train set

        self.train_dataset = self._build_dataset(corpus_train, self.src_charset, self.trg_charset)  # Build train dataset
        self.test_dataset = self._build_dataset(corpus_test, self.src_charset, self.trg_charset)  # Build test dataset
        self.valid_dataset = self._build_dataset(corpus_valid, self.src_charset, self.trg_charset)  # Build validation dataset

    def _build_dataset(self, data, src_charset, trg_charset):
        src_char2idx = src_charset.char2index  # Get source character to index mapping
        trg_char2idx = trg_charset.char2index  # Get target character to index mapping

        input_tensor = torch.zeros((len(data[0]), MAX_SEQUENCE_LENGTH), dtype=torch.long)  # Initialize input tensor
        output_tensor = torch.zeros((len(data[1]), MAX_SEQUENCE_LENGTH), dtype=torch.long)  # Initialize output tensor

        for i in range(len(data)):
            in_seq = data[0][i].lower().ljust(MAX_SEQUENCE_LENGTH-1, '#') + '$'  # Pad and add EOS token to source sequence
            out_seq = data[1][i].ljust(MAX_SEQUENCE_LENGTH-1, '#') + '$'  # Pad and add EOS token to target sequence

            for j, char in enumerate(in_seq):
                input_tensor[i, j] = src_char2idx.get(char)  # Convert source characters to indices

            for j, char in enumerate(out_seq):
                output_tensor[i, j] = trg_char2idx.get(char)  # Convert target characters to indices

        return TensorDataset(input_tensor, output_tensor)  # Return a TensorDataset with input and output tensors

    def get_random_sample(self):
        return self.__getitem__(np.random.randint(len(self.src_words)))  # Return a random sample from the dataset