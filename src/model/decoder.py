import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from ..utils.config import *

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size ,
                       cell_type = 'gru', num_layers = 1,
                       bidirectional =False, use_attention =  False,
                       dropout = 0, device=None):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers  
        self.device = device  
        self.use_attention = use_attention    
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        self.attention = Attention(self.hidden_size)
        
        if self.cell_type == 'gru':
            self.dec_rnn = nn.GRU(input_size= self.hidden_size,
                          hidden_size= self.hidden_size,
                          num_layers= self.num_layers,
                          bidirectional= bidirectional, batch_first=True)
        elif self.cell_type == 'lstm':
            self.dec_rnn = nn.LSTM(input_size= self.hidden_size,
                          hidden_size= self.hidden_size,
                          num_layers= self.num_layers,
                          bidirectional= bidirectional, batch_first=True)
        else:
            self.dec_rnn = nn.RNN(input_size= self.hidden_size,
                          hidden_size= self.hidden_size,
                          num_layers= self.num_layers,
                          bidirectional= bidirectional, batch_first=True)
        
        self.out = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(0, MAX_SEQUENCE_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            
            if self.use_attention:           
                attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        if self.use_attention:
            attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, data, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(data))
        if self.use_attention:
            query = hidden.permute(1, 0, 2)
            context, attn_weights = self.attention(query, encoder_outputs)
            input_rnn = torch.cat((embedded, context), dim=2)
        else:
            attn_weights = None
            input_rnn = embedded
        output, hidden = self.dec_rnn(input_rnn, hidden)
        output = self.out(output)
        return output, hidden, attn_weights