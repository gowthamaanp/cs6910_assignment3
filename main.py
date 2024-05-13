from src.train import *
from src.inference import *

network_config = {
    'embedding_size': 64,
    'en_layers': 1,
    'de_layers': 1,
    'hidden_size': 16,
    'cell': 'gru',
    'bidirectional': False,
    'dropout': 0.0,
    'beam_size': 5,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 30,
    'lang': 'tam',
    'use_attention': False
}

# encoder, decoder, training_loss, validation_loss = train(network_config)

# torch.save(encoder, "model/outputs/encoder.h5")
# torch.save(decoder, "model/outputs/decoder.h5")

tensor, word, _ = inference("adangoiah", network_config['lang'])
print(tensor)

print(word)