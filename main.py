import argparse

from src.train import train
from src.inference import inference
from src.test import test
from src.sweep import sweep

def main(command_line=None):
    
    WANDB_PROJECT = "CS6910_AS3"
    WANDB_ENTITY = "ed23s037"
    
    default_train_config = {
        'embedding_size': 64,
        'num_layers': 2,
        'hidden_size': 1024,
        'cell': 'gru',
        'bidirectional': False,
        'dropout': 0.4,
        'beam_width': 3,
        'learning_rate': 1e-3,
        'batch_size': 256,
        'epochs': 15,
        'lang': 'tam',
        'use_attention': False
    }

    parser = argparse.ArgumentParser(description="Neural Transliteration")
    subparsers = parser.add_subparsers(help="Train, Evaluate or Infer the Encoder-Decoder RNN Model", dest="func")
    parser.add_argument("-wp", "--wandb_project", type=str, default=WANDB_PROJECT, help="Wandb project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default=WANDB_ENTITY, help="Wandb entity name")
    
    parser_train = subparsers.add_parser("train", help="Train the model", description="Train the model")
    parser_train.add_argument("-ep", "--epochs", metavar='', type=int, default=10, help="Number of epochs")
    parser_train.add_argument("-bs", "--batch_size", metavar='', type=int, default=256, help="Batch size")
    parser_train.add_argument("-lr", "--learning_rate", metavar='', type=float, default=1e-3, help="Learning rate")
    parser_train.add_argument("-es", "--embedding_size", metavar='', type=int, default=32, help="Input Embedding Size")
    parser_train.add_argument("-nl", "--num_layers", metavar='', type=int, default=2, help="Number of layers in the encoder and decoder")
    parser_train.add_argument("-hs", "--hidden_size", metavar='', type=int, default=1024, help="Number of hidden layers in the model")
    parser_train.add_argument("-ce", "--cell", metavar='', type=str, default="gru", help="Type of RNN Cell to be used in the model (RNN, GRU, LSTM)")
    parser_train.add_argument("-bi", "--bidirectional", metavar='', type=lambda x: (str(x).lower() == 'true'), default=False, help="Use bidirectional encoding in the Encoder",)
    parser_train.add_argument("-dr", "--dropout", metavar='',type=float, default=0.3, help="Dropout probability")
    parser_train.add_argument("-bm", "--beam_width", metavar='', type=int, default=3, help="Beam width for Beam Search Decoding")
    parser_train.add_argument("-ln", "--lang", metavar='', type=str, default="tam", help="Language of the Transliteration taks(eng -> lang)")
    parser_train.add_argument("-ua", "--use_attention", metavar='', type=lambda x: (str(x).lower() == 'true'), default=False, help="User attention")
    
    parser_eval = subparsers.add_parser("eval", help="Evaluate the trained model", description="Evaluate the model")
    parser_eval.add_argument("-bs", "--batch_size", metavar='', type=int, default=256, help="Batch size")
    parser_eval.add_argument("-ln", "--lang", metavar='', type=str, default="tam", help="Language of the Transliteration taks(eng -> lang)")
    
    parser_infer = subparsers.add_parser("infer", help="Infer the trained model", description="Use/Infer the model")
    parser_infer.add_argument("-iw", "--input_word", metavar='', type=str, default="thamizh", help="Word to transliterate(Input)")
    parser_infer.add_argument("-ln", "--lang", metavar='', type=str, default="tam", help="Language of the Transliteration taks(eng -> lang)")
    
    parser_sweep = subparsers.add_parser("sweep", help="Hyperparameter Sweep using Wandb", description="Hyperparameter Sweep")
    parser_sweep.add_argument("-wp", "--wandb_project", metavar='', type=str, default=WANDB_PROJECT, help="Wandb project name")
    parser_sweep.add_argument("-we", "--wandb_entity", metavar='', type=str, default=WANDB_ENTITY, help="Wandb entity name")
    parser_sweep.add_argument("-ak", "--api_key", metavar='', type=str, help="Wandb API key", required=True)
    
    args = parser.parse_args(command_line)
    default_train_config.update(vars(args))

    if args.func=="train":
        # Training Parameters
        print("Parameters:")
        for key, value in default_train_config.items():
            print(f"{key}: {value}")
        _, _, training_loss, validation_loss, training_accuarcy, validation_accuarcy = train(default_train_config)
        
    elif args.func=="eval":
        ca, wa = test(lang=default_train_config['lang'], batch_size=default_train_config['batch_size'])
        print('Character Accuracy: %.4f Word Accuarcy: %f' % (ca, wa))
        
    elif args.func=="infer":
        _, output_word, _ = inference(args.input_word, default_train_config['lang'])
        print(output_word)
    
    elif args.func=="sweep":
        sweep(api_key=args.api_key, project=args.wandb_project, entity=args.wandb_entity)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()