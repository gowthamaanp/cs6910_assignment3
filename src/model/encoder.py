import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size,
                       cell_type='gru', num_layers=1,
                       bidirectional=False,
                       dropout=0):
        super(Encoder, self).__init__()

        # Initializing parameters
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers

        # Embedding layer to convert input indices to embeddings
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Selecting the appropriate RNN cell type
        if self.cell_type == 'gru':
            self.rnn_cell = nn.GRU(input_size=self.embedding_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=bidirectional, 
                                   batch_first=True)
        elif self.cell_type == 'lstm':
            self.rnn_cell = nn.LSTM(input_size=self.embedding_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=bidirectional, 
                                    batch_first=True)
        else:
            self.rnn_cell = nn.RNN(input_size=self.embedding_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=bidirectional, 
                                   batch_first=True)

    def forward(self, data):
        # Embedding input data
        embedded = self.dropout(self.embedding(data))
        
        # Passing embedded data through the RNN
        output, hidden = self.rnn_cell(embedded)
        
        return output, hidden
