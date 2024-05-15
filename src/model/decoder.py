import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from ..utils.config import *

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size ,
                       cell_type = 'gru', num_layers = 1,
                       use_attention =  False, bidirectional = False,
                       dropout = 0, device = 'cpu', beam_width = 1):
        super(Decoder, self).__init__()
        
        # Initializing parameters
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.num_layers = num_layers  
        self.use_attention = use_attention 
        self.device = device   
        self.beam_width = beam_width
        self.input_size = (3 if self.use_attention and self.bidirectional else 2 if self.use_attention else 1)*self.hidden_size
        
        # Embedding layer to convert input indices to embeddings
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Bahdanau attention head
        self.attention = Attention(self.hidden_size, birectional=self.bidirectional)
        
        # Selecting the appropriate RNN cell type
        if self.cell_type == 'gru':
            self.rnn_cell = nn.GRU(input_size= self.input_size,
                                   hidden_size= self.hidden_size,
                                   num_layers= self.num_layers,
                                   batch_first=True)
        elif self.cell_type == 'lstm':
            self.rnn_cell = nn.LSTM(input_size= self.input_size,
                                    hidden_size= self.hidden_size,
                                    num_layers= self.num_layers,
                                    batch_first=True)
        else:
            self.rnn_cell = nn.RNN(input_size= self.input_size,
                                   hidden_size= self.hidden_size,
                                   num_layers= self.num_layers,
                                   batch_first=True)
        
        # Linear output layer
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # Initialize variables
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        
        if self.beam_width >= 1:
            decoder_outputs = self.beam_search_decoding(encoder_outputs, decoder_hidden)
        
        else:
            # Pass the encoder outputs/hidden state through the decoder
            for i in range(0, MAX_SEQUENCE_LENGTH):
                # Forward pass
                decoder_output, decoder_hidden, attn_weights = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_outputs.append(decoder_output)
                
                if self.use_attention:           
                    attentions.append(attn_weights)
                    
                # Teacher forcing
                if target_tensor is not None:
                    # Teacher forcing: Feed the target as the next input
                    decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()  # detach from history as input
        
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            # Apply softmax to determine the character
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
            
            if self.use_attention:
                attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions
    
    # Performs beam search decoding
    def beam_search_decoding(self, encoder_outputs, encoder_hidden):
        
        batch_size = encoder_outputs.size(0)
        
        # Initialize the beam with the start token and its log probability
        decoder_input = torch.tensor([SOS_TOKEN] * batch_size, device=self.device).unsqueeze(1)
        decoder_hidden = encoder_hidden
        log_probs = torch.zeros(batch_size, 1, device=self.device)
        
        # Initialize the beam with the first sequence
        beam = [(decoder_input, log_probs, decoder_hidden)]
        
        # Iterate over the maximum length
        for _ in range(MAX_SEQUENCE_LENGTH):
            new_beam = []
            
            # Iterate over the current beam
            for seq, log_prob, hidden in beam:
                decoder_output, new_hidden, _ = self.forward_step(seq[:, [-1]], hidden, encoder_outputs)
                log_probs = log_prob + decoder_output.squeeze(1)
                
                # Find the top beam_width candidates
                top_log_probs, top_indices = log_probs.topk(self.beam_width, dim=-1)
                
                # Update the beam with the new candidates
                for new_log_prob, new_index in zip(top_log_probs, top_indices):
                    new_seq = torch.cat([seq, new_index.unsqueeze(1)], dim=1)
                    new_beam.append((new_seq, new_log_prob, new_hidden))
            
            # Sort the beam by log probability and keep the top beam_width sequences
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:self.beam_width]
        
        # Return the decoded sequences
        return [seq.tolist() for seq, _, _ in beam]
    
    def forward_step(self, data, hidden, encoder_outputs):
        # Embedding input data
        embedded =  self.dropout(self.embedding(data))
        
        if self.bidirectional:
            hidden = hidden.sum(dim=0).unsqueeze(0)
        
        # Optional attention usage
        if self.use_attention:
            # Attention implementation
            # Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
            query = hidden.permute(1, 0, 2)
            context, attn_weights = self.attention(query, encoder_outputs)
            input_rnn = torch.cat((embedded, context), dim=2)
        else:
            # Use th embeddings if attention is not used
            attn_weights = None
            input_rnn = embedded
        
        # Passing embedded data through the RNN
        
        output, hidden = self.rnn_cell(input_rnn, hidden)
        
        # Pass the outputs through the linear layer
        output = self.out(output)
        
        return output, hidden, attn_weights
    
    