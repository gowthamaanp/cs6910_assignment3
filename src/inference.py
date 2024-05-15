import torch
import pandas as pd
from .data.charset import CharSet, MAX_SEQUENCE_LENGTH, PAD_TOKEN

# Transliterate a single word from English to Tamil
def inference(word, lang):

    # Get the character set
    src_char2idx = CharSet(language='eng').char2index
    trg_idx2char = CharSet(language=lang).index2char

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the encoder and decoder
    encoder = torch.load(f"./model/outputs/{lang}_encoder.h5").to(device)
    decoder = torch.load(f"./model/outputs/{lang}_decoder.h5").to(device)
    encoder.eval()
    decoder.eval()
    
    # Do a forward pass
    with torch.no_grad():
        input_tensor = input_tensor = torch.zeros((1, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        in_seq = word.lower().ljust(MAX_SEQUENCE_LENGTH-1, '#') + '$'
        for j, char in enumerate(in_seq):
            input_tensor[0, j] = src_char2idx.get(char)
        input_tensor = input_tensor.to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, decoder_attn = decoder(encoder_outputs, encoder_hidden)
        
        # Consider the highest scoring output
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        # Get the encoded word
        encoded_word = []
        for idx in decoded_ids:
            if idx.item() == PAD_TOKEN:
                break
            encoded_word.append(idx.item())
        
        # Convert the encoding to string
        decoded_word = "".join([trg_idx2char.get(char) for char in encoded_word])
    
    # Display the word in a dataframe
    df = pd.DataFrame(columns=['Input', 'Output'])
    df = df._append({'Input': word, 'Output': decoded_word}, ignore_index=True)
    
    return encoded_word, df, decoder_attn 
    