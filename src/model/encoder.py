import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size ,
                       cell_type = 'gru', num_layers = 1,
                       bidirectional =False,
                       dropout = 0, device=None):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        
        self.dropout = nn.Dropout(dropout)

        if self.cell_type == 'gru':
            self.enc_rnn = nn.GRU(input_size= self.embedding_size,
                          hidden_size= self.hidden_size,
                          num_layers= self.num_layers,
                          bidirectional= bidirectional, batch_first=True)
        elif self.cell_type == 'lstm':
            self.enc_rnn = nn.LSTM(input_size= self.embedding_size,
                          hidden_size= self.hidden_size,
                          num_layers= self.num_layers,
                          bidirectional= bidirectional, batch_first=True)
        else:
            self.enc_rnn = nn.RNN(input_size= self.embedding_size,
                          hidden_size= self.hidden_size,
                          num_layers= self.num_layers,
                          bidirectional= bidirectional, batch_first=True)

    def forward(self, data):
        embedded = self.dropout(self.embedding(data))
        output, hidden = self.enc_rnn(embedded)
        return output, hidden