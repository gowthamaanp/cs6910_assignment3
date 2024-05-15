import torch

from src.train import train
from src.inference import inference
from src.test import test

network_config = {
    'embedding_size': 64,
    'en_layers': 2,
    'de_layers': 2,
    'hidden_size': 1024,
    'cell': 'gru',
    'bidirectional': False,
    'dropout': 0.4,
    'beam_size': 5,
    'learning_rate': 1e-3,
    'batch_size': 256,
    'epochs': 15,
    'lang': 'tam',
    'use_attention': False
}

network_config = {
    'embedding_size': 16,
    'en_layers': 1,
    'de_layers': 1,
    'hidden_size': 2,
    'cell': 'gru',
    'bidirectional': True,
    'dropout': 0.4,
    'beam_size': 5,
    'learning_rate': 1e-3,
    'batch_size': 256,
    'epochs': 1,
    'lang': 'tam',
    'use_attention': True
}
encoder, decoder, training_loss, validation_loss, training_accuarcy, validation_accuarcy = train(network_config)

torch.save(encoder, f"model/outputs/{network_config['lang']}_encoder.h5")
torch.save(decoder, f"model/outputs/{network_config['lang']}decoder.h5")

tensor, word, _ = inference("gowthaman", network_config['lang'])

print(test(lang=network_config['lang'], batch_size=network_config['batch_size']))