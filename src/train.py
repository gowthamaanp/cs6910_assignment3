import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from .data.dataset import TransliterationDataset
from .model.encoder import Encoder
from .model.decoder import Decoder
from .utils.helpers import *

def train_epoch(dataloader, encoder, decoder, enc_optimizer, dec_optimizer, criterion, device):
    total_loss = 0
    total_accuracy = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        (input_tensor, target_tensor) = (input_tensor.to(device), target_tensor.to(device))

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        predicted_tensor = torch.argmax(decoder_outputs, dim=2)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
    
        correct = torch.eq(predicted_tensor, target_tensor).float().sum()
        accuracy = correct / len(target_tensor)

        total_loss += loss.item()
        total_accuracy += accuracy.item()
    return total_loss / len(dataloader), total_accuracy/len(dataloader)

def val_epoch(dataloader, encoder, decoder, criterion, device):
    total_loss = 0
    total_accuracy = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        (input_tensor, target_tensor) = (input_tensor.to(device), target_tensor.to(device))
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        predicted_tensor = torch.argmax(decoder_outputs, dim=2)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        correct = torch.eq(predicted_tensor, target_tensor).float().sum()
        accuracy = correct / len(target_tensor)
    
        total_loss += loss.item()
        total_accuracy += accuracy.item()

    return total_loss / len(dataloader), total_accuracy/len(dataloader)

def train(config):
    
    print("Initiating Training...")

    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    embedding_size = config['embedding_size']
    num_layers_encoder = config['en_layers']
    num_layers_decoder = config['de_layers']
    cell_type = config['cell']
    bidirectional = config['bidirectional']
    dropout = config['dropout']
    epochs = config['epochs']
    use_attention = config['use_attention']
    lang = config['lang']
    beam_size = config['beam_size']
    
    print("Loading data...")
    
    device = torch.device("cpu")
    torch.cuda.empty_cache()
    
    dataset = TransliterationDataset(src_lang='eng', trg_lang=lang)    
    train_dataloader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset.valid_dataset, batch_size=batch_size, shuffle=True)  
    
    print("Data loaded ✅")
    
    encoder = Encoder(input_size=dataset.src_charset.get_length(), embedding_size=embedding_size, 
                      hidden_size=hidden_size, cell_type = cell_type, num_layers = num_layers_encoder,
                      bidirectional =bidirectional,
                      dropout = dropout, device='cpu').to(device) 
    
    decoder =  Decoder(output_size=dataset.trg_charset.get_length(), hidden_size=hidden_size,
                       cell_type = cell_type, num_layers = num_layers_decoder,
                       bidirectional =bidirectional, use_attention= use_attention,
                       dropout = dropout, device='cpu').to(device)
    
    log_frequency = 1
    start = time.time()
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    
    loss_train = 0
    loss_valid = 0
    accuracy_train = 0
    accuracy_valid = 0
    
    encoder_optimizer = Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    print("Starting training loop...")
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        loss, accuracy = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        training_loss.append(loss)
        training_accuracy.append(accuracy)
        loss_train += loss
        accuracy_train +=accuracy
        
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            loss, accuracy = val_epoch(valid_dataloader, encoder, decoder, criterion, device)
            validation_loss.append(loss)
            validation_accuracy.append(accuracy)
            loss_valid += loss
            accuracy_valid +=accuracy
        
        if epoch % log_frequency == 0:
            train_loss_avg = loss_train / log_frequency
            valid_loss_avg = loss_valid / log_frequency
            train_accuracy_avg = accuracy_train / log_frequency
            valid_accuracy_avg = accuracy_valid / log_frequency
            print('%s (%d %d%%) Loss T: %.4f V: %.4f Accuracy T: %.4f V: %.4f' % (timeSince(start, epoch / epochs), 
                                    epoch, epoch / epochs * 100, train_loss_avg, valid_loss_avg, train_accuracy_avg, valid_accuracy_avg))
            loss_train = 0
            loss_valid = 0
            accuracy_train = 0
            accuracy_valid = 0
    return encoder, decoder, training_loss, validation_loss