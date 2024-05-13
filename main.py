from src.train import *
from src.inference import *

network_config = {
    'embedding_size': 16,
    'en_layers': 1,
    'de_layers': 1,
    'hidden_size': 10,
    'cell': 'lstm',
    'bidirectional': False,
    'dropout': 0.0,
    'beam_size': 5,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 5,
    'lang': 'tam',
    'use_attention': False
}

encoder, decoder, training_loss, validation_loss = train(network_config)

word, _ = inference("seenivasa", encoder, decoder, network_config['lang'])

print(word)