import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

from .data.charset import PAD_TOKEN
from .data.dataset import TransliterationDataset

tamil_font = FontProperties(fname='font.ttf')

def tensor_row_to_string(num_to_char, tensor_row):
    """
    Convert a tensor row (list of numbers) to a string
    """
    word = ''.join(num_to_char[num.item()] for num in tensor_row if num.item() in num_to_char)
    return word

def save_attention_maps(input_word, output_word, attentions):
    # Convert the attentions tensor to a numpy array
    attentions = attentions.cpu().numpy()
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the heatmap
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticks(range(len(input_word)))
    ax.set_yticks(range(len(output_word)))
    ax.set_xticklabels(list(input_word), rotation=90)
    ax.set_yticklabels(list(output_word), fontproperties=tamil_font)
    # Adjust spacing between ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Save the graph
    filename = f"{input_word.replace(' ', '_')}_{output_word.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_attentions(lang = 'tam'):
    # Prepare testing data
    dataset = TransliterationDataset(src_lang='eng', trg_lang=lang)   
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=10, shuffle=True)
    
    # Load the encoder and decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(f"./model/outputs/{lang}_encoder.h5").to(device)
    decoder = torch.load(f"./model/outputs/{lang}_decoder.h5").to(device)
    encoder.eval()
    decoder.eval()
    
    # Initialize variables
    x = None
    y = None
    y_pred = None
    
    with torch.no_grad():
    # Do a forward pass
        for data in test_dataloader:
            input_tensor, target_tensor = data
            (input_tensor, target_tensor) = (input_tensor.to(device), target_tensor.to(device))
            
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, attention = decoder(encoder_outputs, encoder_hidden)
            predicted_tensor = torch.argmax(decoder_outputs, dim=2)

            x = input_tensor
            y = target_tensor
            y_pred = predicted_tensor
            break
    
    torch.save(attention, 'attention.pt')
    eng_words = [tensor_row_to_string(dataset.src_charset.index2char, row) for row in x]
    tam1_words = [tensor_row_to_string(dataset.trg_charset.index2char, row) for row in y_pred]
    
    for i in range(len(eng_words)):
        save_attention_maps(eng_words[i], tam1_words[i], attention[i])