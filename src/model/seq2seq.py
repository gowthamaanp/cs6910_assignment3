import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers, cell_type='lstm', bidirectional=False, dropout=0.0):
        super(Seq2SeqModel, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_dim)

        if cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout)
            self.decoder = nn.RNN(hidden_dim * (2 if bidirectional else 1), hidden_dim, num_layers, dropout=dropout)
        elif cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout)
            self.decoder = nn.LSTM(hidden_dim * (2 if bidirectional else 1), hidden_dim, num_layers, dropout=dropout)
        elif cell_type == 'gru':
            self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout)
            self.decoder = nn.GRU(hidden_dim * (2 if bidirectional else 1), hidden_dim, num_layers, dropout=dropout)

    def forward(self, input_seq, hidden=None, beam_size=1):
        batch_size = input_seq.size(1)
        seq_length = input_seq.size(0)

        embedded = self.embedding(input_seq)

        outputs, hidden = self.encoder(embedded, hidden)

        if self.bidirectional:
            hidden = (hidden[0].view(self.num_layers, 2, batch_size, self.hidden_dim)[:, :1].contiguous(),
                      hidden[1].view(self.num_layers, 2, batch_size, self.hidden_dim)[:, :1].contiguous())
        else:
            hidden = (hidden[0].view(self.num_layers, 1, batch_size, self.hidden_dim),
                      hidden[1].view(self.num_layers, 1, batch_size, self.hidden_dim))

        if beam_size > 1:
            hidden = (hidden[0].repeat(1, beam_size, 1, 1),
                      hidden[1].repeat(1, beam_size, 1, 1))
            batch_size = batch_size * beam_size

        decoder_input = torch.zeros(1, batch_size, self.hidden_dim * (2 if self.bidirectional else 1), device=input_seq.device)
        decoder_outputs = []

        hypotheses = torch.zeros(batch_size, beam_size, seq_length, device=input_seq.device).long()
        scores = torch.zeros(batch_size, beam_size, device=input_seq.device)

        for t in range(seq_length):
            output, hidden = self.decoder(decoder_input, hidden)
            decoder_output = output
            decoder_outputs.append(decoder_output)

            if beam_size > 1:
                decoder_output = decoder_output.view(batch_size, beam_size, -1)
                log_probs = torch.log_softmax(decoder_output, dim=-1)
                scores = scores.unsqueeze(-1) + log_probs
                scores, indices = scores.view(batch_size, -1).topk(beam_size, dim=-1)
                hypotheses[:, :, t] = indices % self.output_size
                decoder_input = hypotheses[:, :, t].view(1, batch_size * beam_size, -1)
            else:
                decoder_input = output

        if beam_size > 1:
            decoder_outputs = [output.view(batch_size, beam_size, -1) for output in decoder_outputs]
            decoder_outputs = torch.stack(decoder_outputs, dim=2)
            decoder_outputs = decoder_outputs.transpose(1, 2)
        else:
            decoder_outputs = torch.stack(decoder_outputs)

        return decoder_outputs