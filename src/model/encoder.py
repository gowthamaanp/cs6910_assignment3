import torch
import torch.nn as nn
import random

class Encoder(nn.Module):

    def __init__(self, input_dim, embed_dim, hidden_dim ,
                       cell_type = 'gru', layers = 1,
                       bidirectional =False,
                       dropout = 0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.enc_embed_dim = embed_dim
        self.enc_hidden_dim = hidden_dim
        self.enc_rnn_type = cell_type
        self.enc_layers = layers
        self.enc_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(self.input_dim, self.enc_embed_dim)

        if self.enc_rnn_type == "gru":
            self.enc_rnn = nn.GRU(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional, dropout=dropout)
        elif self.enc_rnn_type == "lstm":
            self.enc_rnn = nn.LSTM(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional, dropout=dropout)
        else:
            self.enc_rnn = nn.RNN(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional, dropout=dropout)

    def forward(self, x, hidden = None):
        x = self.embedding(x)
        output, hidden = self.enc_rnn(x)
        return output, hidden