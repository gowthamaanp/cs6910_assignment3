'''
Contains classes fo RNN based trainings with features for
1. Encoder - Decoder architecture
2. Attention Mechanism
3. Teacher forcing
4. Active & Passive beam search for inference

'''
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    '''
    Simple RNN based encoder network
    '''
    def __init__(self, input_dim, embed_dim, hidden_dim ,
                       rnn_type = 'gru', layers = 1,
                       bidirectional =False,
                       dropout = 0, device = "cpu"):
        super(Encoder, self).__init__()

        self.input_dim = input_dim #src_vocab_sz
        self.enc_embed_dim = embed_dim
        self.enc_hidden_dim = hidden_dim
        self.enc_rnn_type = rnn_type
        self.enc_layers = layers
        self.enc_directions = 2 if bidirectional else 1
        self.device = device

        self.embedding = nn.Embedding(self.input_dim, self.enc_embed_dim)

        if self.enc_rnn_type == "gru":
            self.enc_rnn = nn.GRU(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional)
        elif self.enc_rnn_type == "lstm":
            self.enc_rnn = nn.LSTM(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional)
        else:
            raise Exception("unknown RNN type mentioned")

    def forward(self, x, x_sz, hidden = None):
        '''
        x_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad

        Return:
            output: (batch_size, max_length, hidden_dim)
            hidden: (n_layer*num_directions, batch_size, hidden_dim) | if LSTM tuple -(h_n, c_n)

        '''
        batch_sz = x.shape[0]
        # x: batch_size, max_length, enc_embed_dim
        x = self.embedding(x)

        ## pack the padded data
        # x: max_length, batch_size, enc_embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, enc_embed_dim --> hidden from all timesteps
        # hidden: n_layer**num_directions, batch_size, hidden_dim | if LSTM (h_n, c_n)
        output, hidden = self.enc_rnn(x)

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, enc_emb_dim*directions)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # output: batch_size, max_length, hidden_dim
        output = output.permute(1,0,2)

        return output, hidden

    def get_word_embedding(self, x):
        ''' Used for obtaining word embeddings from RNN

        x: (1, sequence_length)
        Return:
            out_embed: (n_layer**num_directions, hidden_dim)
        '''
        x_sz = torch.tensor([len(x)])
        x_ = torch.tensor(x).unsqueeze(0).to(dtype=torch.long)
        # x: 1, max_length, enc_embed_dim
        x = self.embedding(x_)

        ## pack the padded data
        # x: max_length, 1, enc_embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, 1, enc_embed_dim --> hidden from all timesteps
        # hidden: n_layer**num_directions, 1, hidden_dim | if LSTM (h_n, c_n)
        output, hidden = self.enc_rnn(x)

        """
        ## ----- Use each time step to compute embedding

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, enc_emb_dim*directions)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        ## Convert bidirectional size to embed_dim
        # output: max_length, batch_size, enc_emb_dim
        if self.enc_directions == 2:
            d_ = self.enc_embed_dim
            output = output[:,:,:d_] + output[:,:,d_:]

        # out_embed:shp: enc_emb_dim
        out_embed = torch.sum( output.squeeze(1), axis = 0)
        ## Add hidden to embed
        if self.enc_embed_dim == self.enc_hidden_dim:
            if self.enc_rnn_type == "lstm": hidden = hidden[0]
            out_embed = out_embed + torch.sum(hidden.squeeze(1), axis = 0)
        """

        ## ----- use final hidden as embedding

        if self.enc_rnn_type == 'lstm':
            out_embed = hidden[0].squeeze()
        else:
            out_embed = hidden.squeeze()

        return out_embed


class Decoder(nn.Module):
    '''
    Used as decoder stage
    '''
    def __init__(self, output_dim, embed_dim, hidden_dim,
                       rnn_type = 'gru', layers = 1,
                       use_attention = True,
                       enc_outstate_dim = None, # enc_directions * enc_hidden_dim
                       dropout = 0, device = "cpu"):
        super(Decoder, self).__init__()

        self.output_dim = output_dim #tgt_vocab_sz
        self.dec_hidden_dim = hidden_dim
        self.dec_embed_dim = embed_dim
        self.dec_rnn_type = rnn_type
        self.dec_layers = layers
        self.use_attention = use_attention
        self.device = device
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
            raise Exception("unknown RNN type mentioned")

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
        '''
        x: (batch_size, 1, dec_embed_dim) -> after Embedding
        enc_output: batch_size, max_length, enc_hidden_dim *num_directions
        hidden: n_layers, batch_size, hidden_size | if LSTM (h_n, c_n)
        '''

        ## perform addition to calculate the score

        # hidden_with_time_axis: batch_size, 1, hidden_dim
        ## hidden_with_time_axis = hidden.permute(1, 0, 2) ## replaced with below 2lines
        hidden_with_time_axis = torch.sum(hidden, axis=0)

        hidden_with_time_axis = hidden_with_time_axis.unsqueeze(1)

        # score: batch_size, max_length, hidden_dim
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # attention_weights: batch_size, max_length, 1
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, hidden_dim)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        # context_vector: batch_size, 1, hidden_dim
        context_vector = context_vector.unsqueeze(1)

        # attend_out (batch_size, 1, dec_embed_dim + hidden_size)
        attend_out = torch.cat((context_vector, x), -1)

        return attend_out, attention_weights

    def forward(self, x, hidden, enc_output):
        '''
        x: (batch_size, 1)
        enc_output: batch_size, max_length, dec_embed_dim
        hidden: n_layer, batch_size, hidden_size | lstm: (h_n, c_n)
        '''
        if (hidden is None) and (self.use_attention is False):
            raise Exception( "No use of a decoder with No attention and No Hidden")

        batch_sz = x.shape[0]

        if hidden is None:
            # hidden: n_layers, batch_size, hidden_dim
            hid_for_att = torch.zeros((self.dec_layers, batch_sz,
                                    self.dec_hidden_dim )).to(self.device)
        elif self.dec_rnn_type == 'lstm':
            hid_for_att = hidden[0] # h_n
        else:
            hid_for_att = hidden

        # x (batch_size, 1, dec_embed_dim) -> after embedding
        x = self.embedding(x)

        if self.use_attention:
            # x (batch_size, 1, dec_embed_dim + hidden_size) -> after attention
            # aw: (batch_size, max_length, 1)
            x, aw = self.attention( x, hid_for_att, enc_output)
        else:
            x, aw = x, 0

        # passing the concatenated vector to the GRU
        # output: (batch_size, n_layers, hidden_size)
        # hidden: n_layers, batch_size, hidden_size | if LSTM (h_n, c_n)
        output, hidden = self.dec_rnn(x, hidden) if hidden is not None else self.dec_rnn(x)

        # output :shp: (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))

        # output :shp: (batch_size * 1, output_dim)
        output = self.fc(output)

        return output, hidden, aw


class Seq2Seq(nn.Module):
    '''
    Used to construct seq2seq architecture with encoder decoder objects
    '''
    def __init__(self, encoder, decoder, pass_enc2dec_hid=False, dropout = 0, device = "cpu"):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pass_enc2dec_hid = pass_enc2dec_hid
        _force_en2dec_hid_conv = False

        if self.pass_enc2dec_hid:
            assert decoder.dec_hidden_dim == encoder.enc_hidden_dim, "Hidden Dimension of encoder and decoder must be same, or unset `pass_enc2dec_hid`"
        if decoder.use_attention:
            assert decoder.enc_outstate_dim == encoder.enc_directions*encoder.enc_hidden_dim,"Set `enc_out_dim` correctly in decoder"
        assert self.pass_enc2dec_hid or decoder.use_attention, "No use of a decoder with No attention and No Hidden from Encoder"


        self.use_conv_4_enc2dec_hid = False
        if  (
              ( self.pass_enc2dec_hid and
                (encoder.enc_directions * encoder.enc_layers != decoder.dec_layers)
              )
              or _force_en2dec_hid_conv
            ):
            if encoder.enc_rnn_type == "lstm" or encoder.enc_rnn_type == "lstm":
                raise Exception("conv for enc2dec_hid not implemented; Change the layer numbers appropriately")

            self.use_conv_4_enc2dec_hid = True
            self.enc_hid_1ax = encoder.enc_directions * encoder.enc_layers
            self.dec_hid_1ax = decoder.dec_layers
            self.e2d_hidden_conv = nn.Conv1d(self.enc_hid_1ax, self.dec_hid_1ax, 1)

    def enc2dec_hidden(self, enc_hidden):
        '''
        Passing enc hidden as context to dec hidden;
        incase of size mismatch do a 1D convolution (or forcefully)

        enc_hidden: n_layer, batch_size, hidden_dim*num_directions
        TODO: Implement the logic for LSTm bsed model
        '''
        # hidden: batch_size, enc_layer*num_directions, enc_hidden_dim
        hidden = enc_hidden.permute(1,0,2).contiguous()
        # hidden: batch_size, dec_layers, dec_hidden_dim -> [N,C,Tstep]
        hidden = self.e2d_hidden_conv(hidden)

        # hidden: dec_layers, batch_size , dec_hidden_dim
        hidden_for_dec = hidden.permute(1,0,2).contiguous()

        return hidden_for_dec


    def forward(self, src, tgt, src_sz, teacher_forcing_ratio = 0):
        '''
        src: (batch_size, sequence_len.padded)
        tgt: (batch_size, sequence_len.padded)
        src_sz: [batch_size, 1] -  Unpadded sequence lengths
        '''
        batch_size = tgt.shape[0]

        # enc_output: (batch_size, padded_seq_length, enc_hidden_dim*num_direction)
        # enc_hidden: (enc_layers*num_direction, batch_size, hidden_dim)
        enc_output, enc_hidden = self.encoder(src, src_sz)

        if self.pass_enc2dec_hid:
        # dec_hidden: dec_layers, batch_size , dec_hidden_dim
            if self.use_conv_4_enc2dec_hid:
                dec_hidden = self.enc2dec_hidden(enc_hidden)
            else:
                dec_hidden = enc_hidden
        else:
            # dec_hidden -> Will be initialized to zeros internally
            dec_hidden = None

        # pred_vecs: (batch_size, output_dim, sequence_sz) -> shape required for CELoss
        pred_vecs = torch.zeros(batch_size, self.decoder.output_dim, tgt.size(1)).to(self.device)

        # dec_input: (batch_size, 1)
        dec_input = tgt[:,0].unsqueeze(1) # initialize to start token
        pred_vecs[:,1,0] = 1 # Initialize to start tokens all batches
        for t in range(1, tgt.size(1)):
            # dec_hidden: dec_layers, batch_size , dec_hidden_dim
            # dec_output: batch_size, output_dim
            # dec_input: (batch_size, 1)
            dec_output, dec_hidden, _ = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )
            pred_vecs[:,:,t] = dec_output

            # # prediction: batch_size
            prediction = torch.argmax(dec_output, dim=1)

            # Teacher Forcing
            if random.random() < teacher_forcing_ratio:
                dec_input = tgt[:, t].unsqueeze(1)
            else:
                dec_input = prediction.unsqueeze(1)

        return pred_vecs #(batch_size, output_dim, sequence_sz)

    def inference(self, src, max_tgt_sz=50, debug = 0):
        '''
        single input only, No batch Inferencing
        src: (sequence_len)
        '''
        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        # enc_output: (batch_size, padded_seq_length, enc_hidden_dim*num_direction)
        # enc_hidden: (enc_layers*num_direction, batch_size, hidden_dim)
        enc_output, enc_hidden = self.encoder(src_, src_sz)

        if self.pass_enc2dec_hid:
        # dec_hidden: dec_layers, batch_size , dec_hidden_dim
            if self.use_conv_4_enc2dec_hid:
                dec_hidden = self.enc2dec_hidden(enc_hidden)
            else:
                dec_hidden = enc_hidden
        else:
            # dec_hidden -> Will be initialized to zeros internally
            dec_hidden = None

        # pred_arr: (sequence_sz, 1) -> shape required for CELoss
        pred_arr = torch.zeros(max_tgt_sz, 1).to(self.device)
        if debug: attend_weight_arr = torch.zeros(max_tgt_sz, len(src)).to(self.device)

        # dec_input: (batch_size, 1)
        dec_input = start_tok.view(1,1) # initialize to start token
        pred_arr[0] = start_tok.view(1,1) # initialize to start token
        for t in range(max_tgt_sz):
            # dec_hidden: dec_layers, batch_size , dec_hidden_dim
            # dec_output: batch_size, output_dim
            # dec_input: (batch_size, 1)
            dec_output, dec_hidden, aw = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )
            # prediction :shp: (1,1)
            prediction = torch.argmax(dec_output, dim=1)
            dec_input = prediction.unsqueeze(1)
            pred_arr[t] = prediction
            if debug: attend_weight_arr[t] = aw.squeeze(-1)

            if torch.eq(prediction, end_tok):
                break

        if debug: return pred_arr.squeeze(), attend_weight_arr
        # pred_arr :shp: (sequence_len)
        return pred_arr.squeeze().to(dtype=torch.long)

    def active_beam_inference(self, src, beam_width=3, max_tgt_sz=50):
        ''' Active beam Search based decoding
        src: (sequence_len)
        '''
        def _avg_score(p_tup):
            ''' Used for Sorting
            TODO: Dividing by length of sequence power alpha as hyperparam
            '''
            return p_tup[0]

        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        # enc_output: (batch_size, padded_seq_length, enc_hidden_dim*num_direction)
        # enc_hidden: (enc_layers*num_direction, batch_size, hidden_dim)
        enc_output, enc_hidden = self.encoder(src_, src_sz)

        if self.pass_enc2dec_hid:
        # dec_hidden: dec_layers, batch_size , dec_hidden_dim
            if self.use_conv_4_enc2dec_hid:
                init_dec_hidden = self.enc2dec_hidden(enc_hidden)
            else:
                init_dec_hidden = enc_hidden
        else:
            # dec_hidden -> Will be initialized to zeros internally
            init_dec_hidden = None

        # top_pred[][0] = Σ-log_softmax
        # top_pred[][1] = sequence torch.tensor shape: (1)
        # top_pred[][2] = dec_hidden
        top_pred_list = [ (0, start_tok.unsqueeze(0) , init_dec_hidden) ]

        for t in range(max_tgt_sz):
            cur_pred_list = []

            for p_tup in top_pred_list:
                if p_tup[1][-1] == end_tok:
                    cur_pred_list.append(p_tup)
                    continue

                # dec_hidden: dec_layers, 1, hidden_dim
                # dec_output: 1, output_dim
                dec_output, dec_hidden, _ = self.decoder( x = p_tup[1][-1].view(1,1), #dec_input: (1,1)
                                                    hidden = p_tup[2],
                                                    enc_output = enc_output, )

                ## π{prob} = Σ{log(prob)} -> to prevent diminishing
                # dec_output: (1, output_dim)
                dec_output = nn.functional.log_softmax(dec_output, dim=1)
                # pred_topk.values & pred_topk.indices: (1, beam_width)
                pred_topk = torch.topk(dec_output, k=beam_width, dim=1)

                for i in range(beam_width):
                    sig_logsmx_ = p_tup[0] + pred_topk.values[0][i]
                    # seq_tensor_ : (seq_len)
                    seq_tensor_ = torch.cat( (p_tup[1], pred_topk.indices[0][i].view(1)) )

                    cur_pred_list.append( (sig_logsmx_, seq_tensor_, dec_hidden) )

            cur_pred_list.sort(key = _avg_score, reverse =True) # Maximized order
            top_pred_list = cur_pred_list[:beam_width]

            # check if end_tok of all topk
            end_flags_ = [1 if t[1][-1] == end_tok else 0 for t in top_pred_list]
            if beam_width == sum( end_flags_ ): break

        pred_tnsr_list = [t[1] for t in top_pred_list ]

        return pred_tnsr_list

    def passive_beam_inference(self, src, beam_width = 7, max_tgt_sz=50):
        '''
        Passive Beam search based inference
        src: (sequence_len)
        '''
        def _avg_score(p_tup):
            ''' Used for Sorting
            TODO: Dividing by length of sequence power alpha as hyperparam
            '''
            return  p_tup[0]

        def _beam_search_topk(topk_obj, start_tok, beam_width):
            ''' search for sequence with maxim prob
            topk_obj[x]: .values & .indices shape:(1, beam_width)
            '''
            # top_pred_list[x]: tuple(prob, seq_tensor)
            top_pred_list = [ (0, start_tok.unsqueeze(0) ), ]

            for obj in topk_obj:
                new_lst_ = list()
                for itm in top_pred_list:
                    for i in range(beam_width):
                        sig_logsmx_ = itm[0] + obj.values[0][i]
                        seq_tensor_ = torch.cat( (itm[1] , obj.indices[0][i].view(1) ) )
                        new_lst_.append( (sig_logsmx_, seq_tensor_) )

                new_lst_.sort(key = _avg_score, reverse =True)
                top_pred_list = new_lst_[:beam_width]
            return top_pred_list

        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        enc_output, enc_hidden = self.encoder(src_, src_sz)

        if self.pass_enc2dec_hid:
        # dec_hidden: dec_layers, batch_size , dec_hidden_dim
            if self.use_conv_4_enc2dec_hid:
                dec_hidden = self.enc2dec_hidden(enc_hidden)
            else:
                dec_hidden = enc_hidden
        else:
            # dec_hidden -> Will be initialized to zeros internally
            dec_hidden = None

        # dec_input: (1, 1)
        dec_input = start_tok.view(1,1) # initialize to start token


        topk_obj = []
        for t in range(max_tgt_sz):
            dec_output, dec_hidden, aw = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )

            ## π{prob} = Σ{log(prob)} -> to prevent diminishing
            # dec_output: (1, output_dim)
            dec_output = nn.functional.log_softmax(dec_output, dim=1)
            # pred_topk.values & pred_topk.indices: (1, beam_width)
            pred_topk = torch.topk(dec_output, k=beam_width, dim=1)

            topk_obj.append(pred_topk)

            # dec_input: (1, 1)
            dec_input = pred_topk.indices[0][0].view(1,1)
            if torch.eq(dec_input, end_tok):
                break

        top_pred_list = _beam_search_topk(topk_obj, start_tok, beam_width)
        pred_tnsr_list = [t[1] for t in top_pred_list ]

        return pred_tnsr_list

##----------------- Simple Correction Networks ---------------------------------
'''
Collection of Second waffer networks to improve hte prediction by corrective measures
'''
class EmbedSeqNet(nn.Module):
    ''' Network for for predicting embedding;
    Trained with character level multinominal training
    '''
    def __init__(self, voc_dim, embed_dim, hidden_dim ,
                       rnn_type = 'gru', layers = 1,
                       bidirectional = True,
                       dropout = 0, device = "cpu"):
        super(EmbedSeqNet, self).__init__()

        self.voc_dim = voc_dim #src_vocab_sz
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.layers = layers
        self.directions = 2 if bidirectional else 1
        self.device = device

        self.embedding = nn.Embedding(self.voc_dim, self.embed_dim)

        if self.rnn_type == "gru":
            self.corr_rnn = nn.GRU(input_size= self.embed_dim,
                          hidden_size= self.hidden_dim,
                          num_layers= self.layers,
                          bidirectional= bidirectional)
        elif self.rnn_type == "lstm":
            self.corr_rnn = nn.LSTM(input_size= self.embed_dim,
                          hidden_size= self.hidden_dim,
                          num_layers= self.layers,
                          bidirectional= bidirectional)
        else:
            raise Exception("unknown RNN type mentioned")

        self.ffnn = nn.Sequential(
            nn.Linear(self.hidden_dim * self.directions, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.voc_dim),
            )

    def forward(self, src, tgt, src_sz):
        '''
        src: (batch_size, sequence_len.padded)
        tgt: (batch_size, sequence_len.padded)
        src_sz: [batch_size, 1] -  Unpadded sequence lengths
        '''
        batch_size = src.shape[0]
        # x: batch_size, max_length, embed_dim
        x = self.embedding(src)

        ## pack the padded data
        # x: max_length, batch_size, embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, src_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, embed_dim
        # _(hidden): n_layer, batch_size, hidden_dim*num_directions | if LSTM (h_n, c_n)
        output, _ = self.corr_rnn(x)

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, embed_dim
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # output: batch_size, mx_seq_length, embed_dim
        output = output.permute(1,0,2)

        #output :shp: batch_size, mx_seq_length, voc_dim
        output = self.ffnn(output)

        #output :shp: batch_size, voc_dim, mx_seq_length
        output = output.permute(0,2,1)

        #predict_vecs: batch_size, voc_dim, max_length
        predict_vecs = torch.zeros(batch_size, self.voc_dim, tgt.size(1) ).to(self.device)
        # sometimes the output size crosses max_seq_size of target
        curr_sz = min( output.shape[2], tgt.size(1) )
        predict_vecs[:,:,:curr_sz] = output[:,:,:curr_sz]

        return predict_vecs

    def inference(self, src):
        '''
        src: (sequence_length)
        '''
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0).to(dtype=torch.long)

        # x: 1, max_length, embed_dim
        x = self.embedding(src_)

        # x: max_length, 1, embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, src_sz, enforce_sorted=False) # unpad

        # output: packed_size, 1, embed_dim
        output, hidden = self.corr_rnn(x)

        # output: max_length, 1, embed_dim
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # output: 1, mx_seq_length, embed_dim
        output = output.permute(1,0,2)
        # output: embed_dim
        output = torch.sum( output.squeeze(0), axis = 0)

        return output


class VocabCorrectorNet(nn.Module):
    '''
    Word predictor (multinominal classification) based on char-seq input
    '''
    def __init__(self, input_dim, output_dim, char_embed_dim, hidden_dim,
                       mode = "multinominal",
                       rnn_type = 'gru', layers = 1,
                       bidirectional = True,
                       dropout = 0, device = "cpu"):

        '''
        modes: {'multinominal', 'embedding'} !Unused!
        '''
        super(VocabCorrectorNet, self).__init__()

        self.input_dim = input_dim #char_vocab_sz
        self.output_dim = output_dim #word_vocab_sz
        self.char_embed_dim = char_embed_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.rnn_type = rnn_type
        self.layers = layers
        self.directions = 2 if bidirectional else 1
        self.device = device

        self.embedding = nn.Embedding(self.input_dim, self.char_embed_dim)

        if self.rnn_type == "gru":
            self.corr_rnn = nn.GRU(input_size= self.char_embed_dim,
                          hidden_size= self.hidden_dim,
                          num_layers= self.layers,
                          bidirectional= bidirectional)
        elif self.rnn_type == "lstm":
            self.corr_rnn = nn.LSTM(input_size= self.char_embed_dim,
                          hidden_size= self.hidden_dim,
                          num_layers= self.layers,
                          bidirectional= bidirectional)
        else:
            raise Exception("unknown RNN type mentioned")

        self.ffnn_multinominal = nn.Sequential(
                nn.Linear(self.hidden_dim * self.directions, self.char_embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.char_embed_dim, self.output_dim),
                )

    def forward(self, src, src_sz):
        '''
        src: (batch_size, sequence_len.padded)
        tgt: (batch_size, sequence_len.padded)
        src_sz: [batch_size, 1] -  Unpadded sequence lengths
        '''
        batch_size = src.shape[0]
        # x: batch_size, max_length, embed_dim
        x = self.embedding(src)

        ## pack the padded data
        # x: batch_size, max_length, embed_dim -> for pack_pad
        x = nn.utils.rnn.pack_padded_sequence(x, src_sz, enforce_sorted=False,
                                                        batch_first= True) # unpad

        # output_: batch_size, packed_size, embed_dim
        # hidden:  n_layer*num_directions, batch_size, hidden_dim | if LSTM (h_n, c_n)
        output_, hidden = self.corr_rnn(x)

        # hidden: n_layer*num_directions, batch_size, hidden_dim | if LSTM h_n
        hidden = hidden if self.rnn_type != "lstm" \
                        else hidden[0] #h_n

        # hidden: 1, batch_size, hidden_dim * directions ->tking only last two layers
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = -1) if self.directions == 2 \
                        else hidden[:,-1,:]

        #output :shp: batch_size, word_voc_dim
        output = self.ffnn_multinominal(hidden.reshape(batch_size, -1))

        return output

    def inference(self, src):
        '''
        src: (sequence_length) TODO
        '''
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0).to(dtype=torch.long)

        # x: 1, max_length, embed_dim
        x = self.embedding(src_)

        ## pack the padded data
        # x: 1, max_length, embed_dim -> for pack_pad
        x = nn.utils.rnn.pack_padded_sequence(x, src_sz, enforce_sorted=False,
                                                        batch_first= True) # unpad

        # output_: 1, packed_size, embed_dim
        # hidden: 1, n_layer*num_directions,  hidden_dim | if LSTM (h_n, c_n)
        output_, hidden = self.corr_rnn(x)

        # hidden: 1, n_layer*num_directions, hidden_dim | if LSTM h_n
        hidden = hidden if self.rnn_type != "lstm" \
                        else hidden[0] #h_n

        # hidden: 1, batch_size, hidden_dim * directions ->tking only last two layers
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = -1) if self.directions == 2 \
                        else hidden[:,-1,:]

        #output :shp: 1, word_voc_dim
        output = self.ffnn_multinominal(hidden)
        output = torch.argmax(output, dim=1)

        return output