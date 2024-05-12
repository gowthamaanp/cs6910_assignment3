def train(config):

    learn_rate = config['learn_rate']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    embedding_size = config['embedding_size']
    num_layers_encoder = config['num_layers_encoder']
    num_layers_decoder = config['num_layers_decoder']
    cell_type = config['cell_type']
    bidirectional = config['bidirectional']
    dropout = config['dropout']
    teach_ratio = config['teach_ratio']
    epochs = config['epochs']
    attention = config['attention']

    input_len = ipLang.n_chars
    output_len = opLang.n_chars
    
    encoder = EncoderRNN(input_len, hidden_size, embedding_size, 
                 num_layers_encoder, cell_type,
                  bidirectional, dropout, batch_size)
    
    train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valData, batch_size=batch_size, shuffle=True)

    encoder_optimizer=optim.Adam(encoder.parameters(),learn_rate)
    decoder_optimizer=optim.Adam(decoder.parameters(),learn_rate)
    loss_fun=nn.CrossEntropyLoss(reduction="sum")

    encoder.to(device)
    seq_len = 0

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0

    for i in range(epochs):
        
        running_loss = 0.0
        train_correct = 0

        encoder.train()
        decoder.train()

        for j,(train_x,train_y) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            train_x=train_x.T
            train_y=train_y.T
            # print("train_x.shapetrain_x.shape)
            seq_len = len(train_y)
            encoder_hidden=encoder.initHidden()
            # for LSTM encoder_hidden shape ((num_layers * num_directions, batch_size,hidden_size),(self.num_layers * num_directions, batch_size, hidden_size))
            encoder_output,encoder_hidden = encoder(train_x,encoder_hidden)
            # encoder_hidden shape (num_layers, batch_size, hidden_size)
            
            
            # lets move to the decoder
            decoder_input = train_y[0] # shape (1, batch_size)
           
            # Handle different numbers of layers in the encoder and decoder
            if num_layers_encoder != num_layers_decoder:
                if num_layers_encoder < num_layers_decoder:
                    remaining_layers = num_layers_decoder - num_layers_encoder
                    # Copy all encoder hidden layers and then repeat the top layer
                    if cell_type == "LSTM":
                        top_layer_hidden = (encoder_hidden[0][-1].unsqueeze(0), encoder_hidden[1][-1].unsqueeze(0))
                        extra_hidden = (top_layer_hidden[0].repeat(remaining_layers, 1, 1), top_layer_hidden[1].repeat(remaining_layers, 1, 1))
                        decoder_hidden = (torch.cat((encoder_hidden[0], extra_hidden[0]), dim=0), torch.cat((encoder_hidden[1], extra_hidden[1]), dim=0))
                    else:
                        top_layer_hidden = encoder_hidden[-1].unsqueeze(0) #top_layer_hidden shape (1, batch_size, hidden_size)
                        extra_hidden = top_layer_hidden.repeat(remaining_layers, 1, 1)
                        decoder_hidden = torch.cat((encoder_hidden, extra_hidden), dim=0)
  
                else:
                    # Slice the hidden states of the encoder to match the decoder layers
                    if cell_type == "LSTM":
                        decoder_hidden = (encoder_hidden[0][-num_layers_decoder:], encoder_hidden[1][-num_layers_decoder:])
                    else :
                        decoder_hidden = encoder_hidden[-num_layers_decoder:]
            else:
                decoder_hidden = encoder_hidden
            
            loss = 0
            correct = 0
           
            for k in range(0, len(train_y)-1):
                
                if attention == "Yes":
                    decoder_output, decoder_hidden, atten_weights = decoder(decoder_input, decoder_hidden, encoder_output)
                else:
                    decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden) # decoder_output shape (1, batch_size, output_size)

                max_prob, index = decoder_output.topk(1) # max_prob shape (1, batch_size, 1)
                index = torch.squeeze(index) # shape (batch_size)
                decoder_output = torch.squeeze(decoder_output)
                loss += loss_fun(decoder_output, train_y[k+1].long())
                
                correct += (index == train_y[k+1]).sum().item()

                # Apply teacher forcing
                use_teacher_forcing = True if random.random() < teach_ratio else False

                if use_teacher_forcing:
                    decoder_input = train_y[k+1]
                
                else:
                    decoder_input = index

            running_loss += loss.item()
            train_correct += correct
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        

        # find train loss and accuracy and print + log to wandb
        if attention == "Yes":
            _, train_accuracy,_, _ = evaluate(trainData,encoder, decoder,output_len,batch_size,hidden_size,num_layers_encoder,num_layers_decoder, cell_type, attention)
        else:
            _, train_accuracy,_= evaluate(trainData,encoder, decoder,output_len,batch_size,hidden_size,num_layers_encoder,num_layers_decoder, cell_type, attention)
        
        print(f"epoch {i}, training loss {running_loss/(len(trainData)* seq_len)}, training accuracy {train_accuracy}")
        if sweeps:
            wandb.log({"epoch": i, "train_loss": running_loss/(len(trainData)* seq_len), "train_accuracy": train_accuracy})
        
        # # find validation loss and accuracy and print + log to wandb
        if attention == "Yes":
            val_loss, val_accuracy,_, _ = evaluate(valData,encoder, decoder,output_len,batch_size,hidden_size,num_layers_encoder,num_layers_decoder, cell_type, attention)
        else:
            val_loss, val_accuracy,_ = evaluate(valData,encoder, decoder,output_len,batch_size,hidden_size,num_layers_encoder,num_layers_decoder, cell_type, attention)
        
        print(f"epoch {i}, validation loss {val_loss}, validation accuracy {val_accuracy}")
        if sweeps:
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model weights
            torch.save(encoder.state_dict(), 'best_encoder.pt')
            torch.save(decoder.state_dict(), 'best_decoder.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break
    return pred
           