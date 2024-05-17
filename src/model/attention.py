import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Simple Bahdanau attention
Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''

class Attention(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(Attention, self).__init__()
        # Linear transformations for query, keys, and combining them
        self.W = nn.Linear(hidden_size, hidden_size)  # Transforming query
        self.U = nn.Linear((2 if bidirectional else 1)*hidden_size, hidden_size)  # Transforming keys
        self.V = nn.Linear(hidden_size, 1)            # Combining query and keys

    def forward(self, query, keys):
        # Calculating attention scores
        scores = self.V(torch.tanh(self.W(query) + self.U(keys)))
        # Squeezing unnecessary dimension and adding a new dimension
        scores = scores.squeeze(2).unsqueeze(1)
        # Applying softmax to obtain attention weights
        weights = F.softmax(scores, dim=-1)
        # Calculating context vector by weighted sum of keys
        context = torch.bmm(weights, keys)
        
        return context, weights
