import torch
from torch.utils.data import DataLoader

from .data.dataset import TransliterationDataset


def test(lang, batch_size):
    dataset = TransliterationDataset(src_lang='eng', trg_lang=lang)   
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cpu")
    encoder = torch.load("./model/outputs/encoder.h5")
    decoder = torch.load("./model/outputs/decoder.h5")
    encoder.eval()
    decoder.eval()
    
    y = []
    y_pred = []
    
    for data in test_dataloader:
        input_tensor, target_tensor = data
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
        predicted_tensor = torch.argmax(decoder_outputs, dim=2)
        
        y.extend(target_tensor)
        y_pred.extend(predicted_tensor)
    
    y = torch.cat(y, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    correct = torch.eq(y_pred, y).float().sum()
    char_accuracy = correct / len(y)
    
    word_accuracy = ((y==y_pred).sum(dim=0)==y.size(1)-1).sum()
    
    return char_accuracy.item()*100, word_accuracy.item()*100