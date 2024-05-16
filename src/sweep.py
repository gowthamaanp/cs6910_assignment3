import wandb
from .train import train

# Sweep Runner
def wandb_sweep_runner():
    # Init sweep run
    with wandb.init() as run:
        # Get training config
        config = wandb.config
        # Sweep run name
        run.name = f"ce_{config['cell']}_nl_{config['num_layers']}_lr_{config['learning_rate']}_bs_{config['batch_size']}_hs_{config['hidden_size']}"
        # Train model
        encoder, decoder, _, _, _, _ = train(config=config, is_sweep=True)
        del encoder
        del decoder

# Wandb Sweep Root Function
def sweep(api_key, project, entity):
    # Sweep Config
    sweep_config = {
        'method': 'bayes',
        'name': 'Q2_SWEEP_1',
        'metric': {
            'name': "val_word_accuracy",
            'goal': 'maximize',
        },
        'parameters': {
            'embedding_size': {'values': [16, 32, 64, 128, 256]},
            'num_layers': {'values': [1,2,3, 4]},
            'hidden_size': {'values': [16, 32, 64, 128, 256, 512, 1024]},
            'cell': {'values': ['lstm', 'gru', 'rnn']},
            'bidirectional': {'values': [False, True]},
            'beam_width': {'values': [1,2,3,4]},
            'dropout': {'values': [0.2, 0.3, 0.4, 0.5]},
            'learning_rate': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
            'batch_size': {'values': [16, 32, 64, 128, 256, 512]},
            "epochs": {'values': [5, 10, 15, 20]},
            'lang': {'value': 'tam'},
            'use_attention': {'value': False},
        },
    }
    # Wandb login
    wandb.login(key=api_key)
    # Initiate sweep
    wandb_id = wandb.sweep(sweep_config, project=project)
    # Run sweep
    wandb.agent(wandb_id, function=wandb_sweep_runner, count=300)
    # Finish sweep
    wandb.finish()