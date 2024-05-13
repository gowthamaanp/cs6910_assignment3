import torch
import torch.nn as nn
import random


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim,
                       rnn_type = 'gru', layers = 1,
                       use_attention = True,
                       enc_outstate_dim = None,
                       dropout = 0):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.dec_hidden_dim = hidden_dim
        self.dec_embed_dim = embed_dim
        self.dec_rnn_type = rnn_type
        self.dec_layers = layers
        self.use_attention = use_attention
        
        if self.use_attention:
            self.enc_outstate_dim = enc_outstate_dim if enc_outstate_dim else hidden_dim
        else:
            self.enc_outstate_dim = 0


        self.embedding = nn.Embedding(self.output_dim, self.dec_embed_dim)

        if self.dec_rnn_type == 'gru':
            self.dec_rnn = nn.GRU(input_size= self.dec_embed_dim + self.enc_outstate_dim, # to concat attention_output
                          hidden_size= self.dec_hidden_dim, # previous Hidden
                          num_layers= self.dec_layers,
                          batch_first = True )
        elif self.dec_rnn_type == "lstm":
            self.dec_rnn = nn.LSTM(input_size= self.dec_embed_dim + self.enc_outstate_dim, # to concat attention_output
                          hidden_size= self.dec_hidden_dim, # previous Hidden
                          num_layers= self.dec_layers,
                          batch_first = True )
        else:
            self.dec_rnn = nn.RNN(input_size= self.dec_embed_dim + self.enc_outstate_dim, # to concat attention_output
                          hidden_size= self.dec_hidden_dim, # previous Hidden
                          num_layers= self.dec_layers,
                          batch_first = True )

        self.fc = nn.Sequential(
            nn.Linear(self.dec_hidden_dim, self.dec_embed_dim), nn.LeakyReLU(),
            # nn.Linear(self.dec_embed_dim, self.dec_embed_dim), nn.LeakyReLU(), # removing to reduce size
            nn.Linear(self.dec_embed_dim, self.output_dim),
            )

        ##----- Attention ----------
        if self.use_attention:
            self.W1 = nn.Linear( self.enc_outstate_dim, self.dec_hidden_dim)
            self.W2 = nn.Linear( self.dec_hidden_dim, self.dec_hidden_dim)
            self.V = nn.Linear( self.dec_hidden_dim, 1)

    def attention(self, x, hidden, enc_output):
        hidden_with_time_axis = torch.sum(hidden, axis=0)
        hidden_with_time_axis = hidden_with_time_axis.unsqueeze(1)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = torch.softmax(self.V(score), dim=1)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        context_vector = context_vector.unsqueeze(1)
        attend_out = torch.cat((context_vector, x), -1)
        return attend_out, attention_weights

    def forward(self, x, hidden, enc_output):
        if (hidden is None) and (self.use_attention is False):
            raise Exception( "No use of a decoder with No attention and No Hidden")
        batch_sz = x.shape[0]

        if hidden is None:
            hid_for_att = torch.zeros((self.dec_layers, batch_sz,
                                    self.dec_hidden_dim )).to(self.device)
        elif self.dec_rnn_type == 'lstm':
            hid_for_att = hidden[0] # h_n
        else:
            hid_for_att = hidden
        x = self.embedding(x)

        if self.use_attention:
            x, aw = self.attention( x, hid_for_att, enc_output)
        else:
            x, aw = x, 0
        output, hidden = self.dec_rnn(x, hidden) if hidden is not None else self.dec_rnn(x)

        output =  output.view(-1, output.size(2))

        output = self.fc(output)

        return output, hidden, aw
