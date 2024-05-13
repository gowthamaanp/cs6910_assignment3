import argparse
import torch

WANDB_PROJECT = "CS6910_AS1"
WANDB_ENTITY = "ed23s037"

network_config = {
    'embedding_size': 16,
    'en_layers': 3,
    'de_layers': 1,
    'hidden_size': 1,
    'cell': 2,
    'bidirectional': 512,
    'dropout': 10,
    'beam_search': False,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 10,
    'lang': 'tam',
    'teach_ratio': 0,
    'attention': False
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-wp",
    "--wandb_project",
    type=str,
    default=WANDB_PROJECT,
    help="Wandb project name",
    required=True,
)
parser.add_argument(
    "-we", "--wandb_entity", type=str, default=WANDB_ENTITY, help="Wandb entity name", required=True
)

def train(config):
    pass

args = parser.parse_args()
network_config.update(vars(args))

# Print the parameters
print("Parameters:")
for key, value in network_config.items():
    print(f"{key}: {value}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, training_loss, training_accuracy, validation_loss, validation_accuracy = train(network_config)

torch.save(model, "models/model.h5")