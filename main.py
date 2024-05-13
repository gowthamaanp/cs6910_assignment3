import torch

from src.train import *
from src.inference import *

network_config = {
    'embedding_size': 32,
    'en_layers': 1,
    'de_layers': 1,
    'hidden_size': 16,
    'cell': 'gru',
    'bidirectional': False,
    'dropout': 0.0,
    'beam_size': 5,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 30,
    'lang': 'tam',
    'use_attention': False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# encoder, decoder, training_loss, validation_loss = train(network_config)

# torch.save(encoder, "encoder.h5")
# torch.save(decoder, "decoder.h5")

tensor, word, _ = inference("seenivasa", network_config['lang'])
print(tensor)

print(word)