import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import wandb
import gc

from .data.dataset import TransliterationDataset
from .model.encoder import Encoder
from .model.decoder import Decoder
from .utils.helpers import *

# Count correctly transliterated words
def count_correct_words(y, y_pred):
    return ((y==y_pred).sum(dim=0)>=y.size(1)-1).sum()

# Single epoch training function
def train_epoch(dataloader, encoder, decoder, enc_optimizer, dec_optimizer, criterion, device):
    total_loss = 0
    total_char_accuracy = 0
    total_word_accuracy = 0
    for data in dataloader:
        # Forward pass
        input_tensor, target_tensor = data
        (input_tensor, target_tensor) = (input_tensor.to(device), target_tensor.to(device))

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        predicted_tensor = torch.argmax(decoder_outputs, dim=2)
        
        # Calculate loss function
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        
        # Back propagation
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()

        # Calculate running character wise accuracy
        correct = torch.eq(predicted_tensor, target_tensor).float().sum()
        accuracy = correct / len(target_tensor)
        
        # Calculate training accuracy
        total_word_accuracy += count_correct_words(target_tensor, predicted_tensor).item()
        total_loss += loss.item()
        total_char_accuracy += accuracy.item()
        
        del input_tensor
        del target_tensor
        
    return total_loss / len(dataloader), total_char_accuracy/len(dataloader), total_word_accuracy/len(dataloader)

# Single epoch validation function
def val_epoch(dataloader, encoder, decoder, criterion, device):
    total_loss = 0
    total_char_accuracy = 0
    total_word_accuracy = 0
    for data in dataloader:
        # Forward pass
        input_tensor, target_tensor = data
        (input_tensor, target_tensor) = (input_tensor.to(device), target_tensor.to(device))
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
        predicted_tensor = torch.argmax(decoder_outputs, dim=2)
        
        # Calculate validation loss
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        
        # Calculate running character wise accuracy
        correct = torch.eq(predicted_tensor, target_tensor).float().sum()
        accuracy = correct / len(target_tensor)
        
        # Calculate validation accuracy
        total_word_accuracy += count_correct_words(target_tensor, predicted_tensor).item()
        total_loss += loss.item()
        total_char_accuracy += accuracy.item()
        
        del input_tensor
        del target_tensor
        
    return total_loss / len(dataloader), total_char_accuracy/len(dataloader), total_word_accuracy/len(dataloader)

def train(config, is_sweep=False):
    
    print("Initiating Training...")
    
    # Training Configuration
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    embedding_size = config['embedding_size']
    num_layers_encoder = config['encoder_layers']
    num_layers_decoder = config['decoder_layers']
    cell_type = config['cell']
    bidirectional = config['bidirectional']
    dropout = config['dropout']
    epochs = config['epochs']
    use_attention = config['use_attention']
    lang = config['lang']
    beam_width = config['beam_width']
    
    print("Loading data...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    
    # Prepare dataset
    dataset = TransliterationDataset(src_lang='eng', trg_lang=lang)    
    train_dataloader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset.valid_dataset, batch_size=batch_size, shuffle=True)  
    
    print("Data loaded ✅")
    
    # Encoder and Decoder model
    encoder = Encoder(input_size=dataset.src_charset.get_length(), embedding_size=embedding_size, 
                      hidden_size=hidden_size, cell_type = cell_type, num_layers = num_layers_encoder,
                      bidirectional =bidirectional,
                      dropout = dropout).to(device) 
    
    decoder =  Decoder(output_size=dataset.trg_charset.get_length(), hidden_size=hidden_size,
                       cell_type = cell_type, num_layers = num_layers_decoder, 
                       bidirectional=bidirectional, use_attention= use_attention, 
                       beam_width=beam_width, dropout = dropout, device = device).to(device)
    
    # Initialize variable
    log_frequency = 1
    start = time.time()
    training_loss = []
    validation_loss = []
    training_char_accuracy = []
    training_word_accuracy = []
    validation_char_accuracy = []
    validation_word_accuracy = []
    
    loss_train = 0
    loss_valid = 0
    accuracy_char_train = 0
    accuracy_char_valid = 0
    accuracy_word_train = 0
    accuracy_word_valid = 0
    
    encoder_optimizer = Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    print("Starting training loop...")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        # Train the model
        loss, char_accuracy, word_accuarcy = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        # Store the loss and accuracy
        training_loss.append(loss)
        training_word_accuracy.append(word_accuarcy)
        training_char_accuracy.append(char_accuracy)
        loss_train += loss
        accuracy_word_train +=word_accuarcy
        accuracy_char_train +=char_accuracy
        
        # Validate the model
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            loss, char_accuracy, word_accuarcy = val_epoch(valid_dataloader, encoder, decoder, criterion, device)
            # Store the loss and accuracy
            validation_loss.append(loss)
            validation_char_accuracy.append(char_accuracy)
            validation_word_accuracy.append(word_accuarcy)
            loss_valid += loss
            accuracy_char_valid +=char_accuracy
            accuracy_word_valid +=word_accuarcy
        
        if is_sweep:
            wandb.log({
                "epochs": epoch,
                "train_loss": training_loss[-1],
                "train_char_accuracy": training_char_accuracy[-1],
                "train_word_accuracy": training_word_accuracy[-1],
                "val_loss": validation_loss[-1],
                "val_char_accuracy": validation_char_accuracy[-1],
                "val_word_accuracy": validation_word_accuracy[-1],
             })
            
            
        # Log training metrics
        if epoch % log_frequency == 0:
            train_loss_avg = loss_train / log_frequency
            valid_loss_avg = loss_valid / log_frequency
            train_accuracy_char_avg = accuracy_char_train / log_frequency
            valid_accuracy_char_avg = accuracy_char_valid / log_frequency
            train_accuracy_word_avg = accuracy_word_train / log_frequency
            valid_accuracy_word_avg = accuracy_word_valid / log_frequency
            print('%s (%d %d%%) Loss T: %.4f V: %.4f Character Accuracy T: %.4f V: %.4f Word Accuracy T: %.4f V: %.4f' % (timeSince(start, epoch / epochs), 
                                    epoch, epoch / epochs * 100, train_loss_avg, valid_loss_avg, train_accuracy_char_avg, valid_accuracy_char_avg, train_accuracy_word_avg, 
                                    valid_accuracy_word_avg))
            loss_train = 0
            loss_valid = 0
            accuracy_char_train = 0
            accuracy_char_valid = 0
            accuracy_word_train = 0
            accuracy_word_valid = 0
    
    if not is_sweep:
        # Save the model after training
        torch.save(encoder, f"./model/outputs/{lang}_encoder.h5")
        torch.save(decoder, f"./model/outputs/{lang}_decoder.h5")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return encoder, decoder, training_loss, validation_loss, [training_char_accuracy, training_word_accuracy], [validation_char_accuracy, validation_word_accuracy]