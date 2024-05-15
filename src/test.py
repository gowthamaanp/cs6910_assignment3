import torch
from torch.utils.data import DataLoader

from .data.dataset import TransliterationDataset

# Testing the trained model
def test(lang, batch_size, save_predictions=False):
    # Prepare testing data
    dataset = TransliterationDataset(src_lang='eng', trg_lang=lang)   
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=batch_size, shuffle=True)
    
    # Load the encoder and decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(f"./model/outputs/{lang}_encoder.h5").to(device)
    decoder = torch.load(f"./model/outputs/{lang}_decoder.h5").to(device)
    encoder.eval()
    decoder.eval()
    
    # Initialize variables
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
            y = torch.cat((y, target_tensor), dim=0)
            y_pred = torch.cat((y_pred, predicted_tensor), dim=0)
        else:
            y = target_tensor
            y_pred = predicted_tensor
        
        # Compute running character wise accuracy
        correct = torch.eq(predicted_tensor, target_tensor).float().sum()
        accuracy = correct / len(target_tensor)

        char_accuracy += accuracy.item()
    
    # Calculate word wise accuarcy
    word_accuracy = ((y==y_pred).sum(dim=0)==y.size(1)-1).sum()
    char_accuracy = char_accuracy/len(test_dataloader)
    
    if save_predictions:
        # save the predictions to a csv file.
        pass
    
    return char_accuracy, word_accuracy.item()*100