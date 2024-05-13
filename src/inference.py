import torch

from .data.charset import CharSet, MAX_SEQUENCE_LENGTH, EOS_TOKEN


def inference(word, lang):

    src_char2idx = CharSet(language='eng').char2index
    trg_idx2char = CharSet(language=lang).index2char

    encoder = torch.load("./model/outputs/encoder.h5")
    decoder = torch.load("./model/outputs/decoder.h5")
    
    with torch.no_grad():
        input_tensor = input_tensor = torch.zeros((1, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        in_seq = word.ljust(MAX_SEQUENCE_LENGTH-1, '#') + '$'
        for j, char in enumerate(in_seq):
            input_tensor[0, j] = src_char2idx.get(char)
            
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        word = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                break
            word.append(idx.item())
        decoded_word = "".join([trg_idx2char.get(char) for char in word])
    return word, decoded_word, decoder_attn 
    