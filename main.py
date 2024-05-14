import torch

from src.train import train
from src.inference import inference
from src.test import test

network_config = {
    'embedding_size': 32,
    'en_layers': 3,
    'de_layers': 3,
    'hidden_size': 1024,
    'cell': 'lstm',
    'bidirectional': False,
    'dropout': 0.3,
    'beam_size': 5,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 15,
    'lang': 'tam',
    'use_attention': False
}

encoder, decoder, training_loss, validation_loss = train(network_config)

torch.save(encoder, "model/outputs/encoder.h5")
torch.save(decoder, "model/outputs/decoder.h5")

tensor, word, _ = inference("gowthaman", network_config['lang'])

print(test(lang=network_config['lang'], batch_size=network_config['batch_size']))