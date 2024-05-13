import torch
from torch.utils.data import DataLoader

from .data.dataset import TransliterationDataset


def test(encoder, decoder, lang, batch_size):
    dataset = TransliterationDataset(src_lang='eng', trg_lang=lang)   
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=batch_size, shuffle=True)
    