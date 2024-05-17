import torch
from torch.utils.data import DataLoader
import pandas as pd

from .data.dataset import TransliterationDataset

def tensor_row_to_string(num_to_char, tensor_row):
    """
    Convert a tensor row (list of numbers) to a string
    """
    word = ''.join(num_to_char[num.item()] for num in tensor_row if num.item() in num_to_char)
    return word

# Testing the trained model
def test(lang, batch_size, save_predictions=True):
    # Prepare testing data
    dataset = TransliterationDataset(src_lang='eng', trg_lang=lang)   
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=batch_size, shuffle=True)
    
    # Load the encoder and decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(f"./model/outputs/{lang}_seq_encoder.h5").to(device)
    decoder = torch.load(f"./model/outputs/{lang}_seq_decoder.h5").to(device)
    encoder.eval()
    decoder.eval()
    
    # Initialize variables
    x = None
    y = None
    y_pred = None
    char_accuracy = 0
    
    # Do a forward pass
    for data in test_dataloader:
        input_tensor, target_tensor = data
        (input_tensor, target_tensor) = (input_tensor.to(device), target_tensor.to(device))
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
        predicted_tensor = torch.argmax(decoder_outputs, dim=2)

        if y is not None:
            x = torch.cat((x, input_tensor), dim=0)
            y = torch.cat((y, target_tensor), dim=0)
            y_pred = torch.cat((y_pred, predicted_tensor), dim=0)
        else:
            x = input_tensor
            y = target_tensor
            y_pred = predicted_tensor
        
        # Compute running character wise accuracy
        correct = torch.eq(predicted_tensor, target_tensor).float().sum()
        accuracy = correct / len(target_tensor)

        char_accuracy += accuracy.item()
    
    # Calculate word wise accuarcy
    word_accuracy = ((y==y_pred).sum(dim=0)>=y.size(1)-1).sum()
    char_accuracy = char_accuracy/len(test_dataloader)
    
    if save_predictions:
        # Assuming your tensor is called 'tensor'
        eng_words = [tensor_row_to_string(dataset.src_charset.index2char, row) for row in x]
        tam_words = [tensor_row_to_string(dataset.trg_charset.index2char, row) for row in y]
        tam1_words = [tensor_row_to_string(dataset.trg_charset.index2char, row) for row in y_pred]

        # Create a pandas DataFrame
        df = pd.DataFrame({
            'Eng': eng_words,
            'Tam': tam_words,
            'Tam1': tam1_words
        })

        # Save the DataFrame to a CSV file
        df.to_csv('tensors_seq.csv', index=False)
    
    return char_accuracy, word_accuracy.item()