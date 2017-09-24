"""
Proposed m-biRNN to learn a self-attention mechanism for prediction of hospitalization
using EHR data based on bi-directional RNN with embedding and the attention and hops mechanism
Author: Jinghe Zhang
"""


import torch
import torch.nn as nn
from torch.autograd import Variable


class Patient2Vec(nn.Module):
    """
    A convolutional embedding layer, then recurrent autoencoder with an encoder, recurrent module, and a decoder.
    In addition, a linear layer is on top of each decode step and the weights are shared at these step.
    """

    def __init__(self, input_size, embed_size, hidden_size, n_layers, n_hops, att_dim, initrange,
                 output_size, rnn_type, seq_len, pad_size, dropout_p=0.5):
        """
        Initilize a recurrent model
        """
        super(Patient2Vec, self).__init__()

        self.initrange = initrange
        # convolution
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_size, stride=2)

        # Embedding
        self.embed = nn.Linear(input_size, embed_size, bias=False)
        # Bidirectional RNN
        self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
                                             batch_first=True, bias=True, bidirectional=False)
        # initialize 2-layer attention weight matrics
        self.att_w1 = nn.Linear(hidden_size * 2, att_dim, bias=False)
        self.att_w2 = nn.Linear(att_dim, n_hops, bias=False)

        # final linear layer
        self.linear = nn.Linear(n_hops * hidden_size * 2, output_size, bias=True)

        self.init_weights()
        self.pad_size = pad_size
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.n_hops = n_hops
        self.func_softmax = nn.Softmax()
        self.func_sigmoid = nn.Sigmoid()
        self.func_tanh = nn.Hardtanh()
        # Add dropout
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

    def init_weights(self):
        """
        weight initialization
        """
        for param in self.parameters():
            param.data.uniform_(-self.initrange, self.initrange)

    def convolutional_layer(self, inputs):
        convolution_all = []
        for i in range(self.seq_len):
            convolution_one_month = []
            for j in range(self.pad_size):
                convolution = self.conv(torch.unsqueeze(inputs[:, i, j], dim=1))
                convolution_one_month.append(convolution)
            convolution_one_month = torch.stack(convolution_one_month)
            convolution_one_month = torch.squeeze(convolution_one_month, dim=3)
            convolution_one_month = torch.transpose(convolution_one_month, 0, 1)
            convolution_one_month = torch.transpose(convolution_one_month, 1, 2)
            vec = torch.bmm(convolution_one_month, inputs[:, i])
            convolution_all.append(vec)
        convolution_all = torch.stack(convolution_all, dim=1)
        convolution_all = torch.squeeze(convolution_all, dim=2)
        return convolution_all

    def embedding_layer(self, convolutions):
        embedding_f = []
        for i in range(self.seq_len):
            embedded = self.embed(convolutions[:, i])
            embedding_f.append(embedded)
        embedding_f = torch.stack(embedding_f)
        embedding_f = torch.transpose(embedding_f, 0, 1)
        embedding_f = self.func_tanh(embedding_f)
        return embedding_f

    def encode_rnn(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_()))
        embedding = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn(embedding, init_state)
        return outputs_rnn

    def add_attention(self, states, batch_size):
        # attention
        alpha = []
        for i in range(self.seq_len):
            m1 = self.att_w1(states[:, i])
            m1_actv = self.func_tanh(m1)
            m2 = self.att_w2(m1_actv)
            alpha.append(m2)
        alpha = torch.stack(alpha)
        alpha = torch.transpose(alpha, 0, 1)
        alpha_actv = []
        for i in range(self.n_hops):
            al_actv = self.func_softmax(alpha[:, :, i])
            alpha_actv.append(al_actv)
        alpha_actv = torch.stack(alpha_actv)
        alpha_actv = torch.transpose(alpha_actv, 0, 1)
        context = torch.bmm(alpha_actv, states)
        context = context.view(batch_size, -1)
        return alpha_actv, context

    def forward(self, inputs, batch_size):
        """
        the recurrent module
        """
        # Convolutional
        convolutions = self.convolutional_layer(inputs)
        # Embedding
        embedding = self.embedding_layer(convolutions)
        # RNN
        states_rnn = self.encode_rnn(embedding, batch_size)
        # Add attentions and get context vector
        alpha, context = self.add_attention(states_rnn, batch_size)
        # alpha = self.add_attention(states_rnn, batch_size)
        # Final linear layer
        linear_y = self.linear(context)
        out = self.func_softmax(linear_y)
        return out, [states_rnn, context, alpha]


class Patient2Vec0(nn.Module):
    """
    A convolutional embedding layer, then recurrent autoencoder with an encoder, recurrent module, and a decoder.
    In addition, a linear layer is on top of each decode step and the weights are shared at these step.
    """

    def __init__(self, input_size, embed_size, hidden_size, n_layers, att_dim, initrange,
                 output_size, rnn_type, seq_len, pad_size, dropout_p=0.5):
        """
        Initilize a recurrent model
        """
        super(Patient2Vec0, self).__init__()

        self.initrange = initrange
        # convolution
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_size, stride=2)

        # Embedding
        self.embed = nn.Linear(input_size, embed_size, bias=False)
        # Bidirectional RNN
        self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
                                         batch_first=True, bias=True, bidirectional=False)
        # initialize 2-layer attention weight matrics
        self.att_w1 = nn.Linear(hidden_size * 2, att_dim, bias=False)

        # final linear layer
        self.linear = nn.Linear(hidden_size * 2, output_size, bias=True)

        self.init_weights()
        self.pad_size = pad_size
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.func_softmax = nn.Softmax()
        self.func_sigmoid = nn.Sigmoid()
        self.func_tanh = nn.Hardtanh()
        # Add dropout
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

    def init_weights(self):
        """
        weight initialization
        """
        for param in self.parameters():
            param.data.uniform_(-self.initrange, self.initrange)

    def convolutional_layer(self, inputs):
        convolution_all = []
        for i in range(self.seq_len):
            convolution_one_month = []
            for j in range(self.pad_size):
                convolution = self.conv(torch.unsqueeze(inputs[:, i, j], dim=1))
                convolution_one_month.append(convolution)
            convolution_one_month = torch.stack(convolution_one_month)
            convolution_one_month = torch.squeeze(convolution_one_month, dim=3)
            convolution_one_month = torch.transpose(convolution_one_month, 0, 1)
            convolution_one_month = torch.transpose(convolution_one_month, 1, 2)
            vec = torch.bmm(convolution_one_month, inputs[:, i])
            convolution_all.append(vec)
        convolution_all = torch.stack(convolution_all, dim=1)
        convolution_all = torch.squeeze(convolution_all, dim=2)
        return convolution_all

    def embedding_layer(self, convolutions):
        embedding_f = []
        for i in range(self.seq_len):
            embedded = self.embed(convolutions[:, i])
            embedding_f.append(embedded)
        embedding_f = torch.stack(embedding_f)
        embedding_f = torch.transpose(embedding_f, 0, 1)
        embedding_f = self.func_tanh(embedding_f)
        return embedding_f

    def encode_rnn(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_()))
        embedding = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn(embedding, init_state)
        return outputs_rnn

    def add_attention(self, states, batch_size):
        # attention
        alpha = []
        for i in range(self.seq_len):
            m1 = self.att_w1(states[:, i])
            m1_actv = self.func_softmax(m1)
            # m2 = self.att_w2(m1_actv)
            alpha.append(m1_actv)
        alpha = torch.stack(alpha)
        alpha = torch.transpose(alpha, 0, 1)
        alpha = torch.transpose(alpha, 1, 2)
        context = torch.bmm(alpha, states)
        context = context.view(batch_size, -1)
        return alpha, context

    def forward(self, inputs, batch_size):
        """
        the recurrent module
        """
        # Convolutional
        convolutions = self.convolutional_layer(inputs)
        # Embedding
        embedding = self.embedding_layer(convolutions)
        # RNN
        states_rnn = self.encode_rnn(embedding, batch_size)
        # Add attentions and get context vector
        alpha, context = self.add_attention(states_rnn, batch_size)
        # alpha = self.add_attention(states_rnn, batch_size)
        # Final linear layer
        linear_y = self.linear(context)
        out = self.func_softmax(linear_y)
        return out, [states_rnn, context, alpha]

# alpha = []
# states_transpose = torch.transpose(states, 1, 2)
# for i in range(seq_len):
#     m1 = model.att_w1(states_transpose[:, :, i])
#     m1_actv = model.func_tanh(m1)
#     m2 = model.att_w2(m1_actv)
#     alpha.append(m2)
# alpha = torch.stack(alpha)
# alpha = torch.transpose(alpha, 0, 1)
# alpha_s = []
# for i in range(n_hops):
#     al = alpha[:, :, i]
#     al_actv = model.func_softmax(al)
#     alpha_s.append(al_actv)
# alpha_s = torch.stack(alpha_s)
# alpha_s = torch.transpose(alpha_s, 0, 1)
#

# class Patient2Vec(nn.Module):
#     """
#     A recurrent autoencoder with an encoder, recurrent module, and a decoder.
#     In addition, a linear layer is on top of each decode step and the weights are shared at these step.
#     """
#
#     def __init__(self, input_size, embed_size, hidden_size, n_layers, n_hops, initrange,
#                  output_size, rnn_type, seq_len, dropout_p=0.5):
#         """
#         Initilize a recurrent autoencoder
#         """
#         super(Patient2Vec, self).__init__()
#
#         self.initrange = initrange
#
#         # Embedding
#         self.embed = nn.Linear(input_size, embed_size, bias=False)
#         # Bidirectional RNN
#         self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
#                                              batch_first=True, bias=True, bidirectional=True)
#         # initialize attention weight matrics
#         self.att_wa = self.init_att_weights((n_hops, hidden_size * 2))
#         self.att_wb = self.init_att_weights((n_hops, hidden_size * 2, hidden_size * 2))
#
#         # final linear layer
#         self.linear = nn.Linear(n_hops * hidden_size * 2, output_size, bias=True)
#
#         self.init_weights()
#         self.input_size = input_size
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.seq_len = seq_len
#         self.n_hops = n_hops
#         self.func_softmax = nn.Softmax()
#         self.func_sigmoid = nn.Sigmoid()
#         self.func_tanh = nn.Hardtanh()
#         # Add dropout
#         self.dropout_p = dropout_p
#         if self.dropout_p > 0:
#             self.dropout = nn.Dropout(p=self.dropout_p)
#
#     def init_att_weights(self, dims):
#         w = np.random.uniform(-self.initrange, self.initrange, dims)
#         att_w = torch.from_numpy(w)
#         att_w = torch.unsqueeze(att_w, 0)
#         att_w = Variable(att_w.float(), requires_grad=True)
#         return att_w
#
#     def init_weights(self):
#         """
#         weight initialization
#         """
#         for param in self.parameters():
#             param.data.uniform_(-self.initrange, self.initrange)
#
#     def embedding_layer(self, inputs):
#         embedding_f = []
#         for i in range(self.seq_len):
#             embedded = self.embed(inputs[:, :, i])
#             embedding_f.append(embedded)
#         embedding_f = torch.stack(embedding_f)
#         embedding_f = torch.transpose(embedding_f, 0, 1)
#         # embedding_f = self.func_tanh(embedding_f)
#         return embedding_f
#
#     def encode_rnn(self, embedding, batch_size):
#         self.weight = next(self.parameters()).data
#         init_state = (Variable(self.weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_()))
#         embedding = self.dropout(embedding)
#         outputs_rnn, states_rnn = self.rnn(embedding, init_state)
#         return outputs_rnn
#
#     def add_visit_attention(self, states, batch_size):
#         # attention
#         atts = []
#         for i in range(batch_size):
#             st = torch.unsqueeze(states[i, :, :], 0)
#             m1 = torch.bmm(self.att_wa, torch.transpose(st, 1, 2))
#             m1_actv = self.func_softmax(torch.squeeze(m1))
#             atts.append(m1_actv)
#         atts = torch.stack(atts)
#         return atts
#
#     def add_var_attention(self, states, batch_size):
#         # attention
#         atts = []
#         for i in range(batch_size):
#             st = torch.unsqueeze(states[i, :, :], 0)
#             atts_each_hop = []
#             for j in range(self.n_hops):
#                 m1 = torch.bmm(self.att_wb[:, j, :, :], torch.transpose(st, 1, 2))
#                 m1 = torch.squeeze(m1)
#                 m1_actv = self.func_tanh(m1)
#                 atts_each_hop.append(m1_actv)
#             atts_each_hop = torch.stack(atts_each_hop)
#             atts.append(atts_each_hop)
#         atts = torch.stack(atts)
#         return atts
#
#     def get_context_vector(self, alpha, beta, states):
#         contexts = []
#         for j in range(self.n_hops):
#             dt = beta[:, j, :, :] * torch.transpose(states, 1, 2)
#             a = torch.unsqueeze(alpha[:, j, :], 2)
#             c = torch.bmm(dt, a)
#             c = torch.squeeze(c)
#             contexts.append(c)
#         contexts = torch.cat(contexts, 1)
#         return contexts
#
#     def forward(self, inputs, batch_size):
#         """
#         the recurrent module
#         """
#         # Embedding
#         embedding = self.embedding_layer(inputs)
#         # RNN
#         states_rnn = self.encode_rnn(embedding, batch_size)
#         # Add attentions
#         alpha = self.add_visit_attention(states_rnn, batch_size)
#         beta = self.add_var_attention(states_rnn, batch_size)
#         # Get context vector
#         contexts = self.get_context_vector(alpha, beta, states_rnn)
#         # Final linear layer
#         linear_y = self.linear(contexts)
#         out = self.func_sigmoid(linear_y)
#         return out, [contexts, alpha, beta]
#         # return states_rnn, alpha, beta


# class Patient2Vec2L(nn.Module):
#     """
#     A recurrent autoencoder with an encoder, recurrent module, and a decoder.
#     In addition, a linear layer is on top of each decode step and the weights are shared at these step.
#     """
#
#     def __init__(self, input_size, embed_size, hidden_size, n_layers, n_hops, att_da, att_db, initrange,
#                  output_size, rnn_type, seq_len, dropout_p=0.5):
#         """
#         Initilize a recurrent autoencoder
#         """
#         super(Patient2Vec2L, self).__init__()
#
#         self.initrange = initrange
#
#         # Embedding
#         self.embed = nn.Linear(input_size, embed_size, bias=False)
#         # Bidirectional RNN
#         self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
#                                              batch_first=True, bias=True, bidirectional=True)
#         # initialize attention weight matrics
#         self.att_w1a = self.init_att_weights((att_da, hidden_size * 2))
#         self.att_w2a = self.init_att_weights((n_hops, att_da))
#
#         self.att_w1b = self.init_att_weights((att_db, hidden_size * 2))
#         self.att_w2b = self.init_att_weights((n_hops, hidden_size * 2, att_db))
#
#         # final linear layer
#         self.linear = nn.Linear(hidden_size * 2, output_size, bias=True)
#
#         self.init_weights()
#         self.input_size = input_size
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.seq_len = seq_len
#         self.n_hops = n_hops
#         self.func_softmax = nn.Softmax()
#         self.func_sigmoid = nn.Sigmoid()
#         self.func_tanh = nn.Hardtanh()
#         # Add dropout
#         self.dropout_p = dropout_p
#         if self.dropout_p > 0:
#             self.dropout = nn.Dropout(p=self.dropout_p)
#
#     def init_att_weights_2d(self, dims):
#         w = np.random.uniform(-self.initrange, self.initrange, dims)
#         att_w = torch.from_numpy(w)
#         att_w = torch.unsqueeze(att_w, 0)
#         att_w = Variable(att_w.float(), requires_grad=True)
#         return att_w
#
#     def init_weights(self):
#         """
#         weight initialization
#         """
#         for param in self.parameters():
#             param.data.uniform_(-self.initrange, self.initrange)
#
#     def embedding_layer(self, inputs):
#         embedding_f = []
#         for i in range(self.seq_len):
#             embedded = self.embed(inputs[:, :, i])
#             embedding_f.append(embedded)
#         embedding_f = torch.stack(embedding_f)
#         embedding_f = torch.transpose(embedding_f, 0, 1)
#         # embedding_f = self.func_tanh(embedding_f)
#         return embedding_f
#
#     def encode_rnn(self, embedding, batch_size):
#         self.weight = next(self.parameters()).data
#         init_state = (Variable(self.weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_()))
#         embedding = self.dropout(embedding)
#         outputs_rnn, states_rnn = self.rnn(embedding, init_state)
#         return outputs_rnn
#
#     def add_visit_attention(self, states, batch_size):
#         # attention
#         atts = []
#         for i in range(batch_size):
#             st = torch.unsqueeze(states[i, :, :], 0)
#             m1 = torch.bmm(self.att_w1a, torch.transpose(st, 1, 2))
#             m1_actv = self.func_tanh(m1)
#             m2 = torch.bmm(self.att_w2a, m1_actv)
#             m2_actv = self.func_softmax(torch.squeeze(m2))
#             atts.append(m2_actv)
#         atts = torch.stack(atts)
#         return atts
#
#     def add_var_attention(self, states, batch_size):
#         # attention
#         atts = []
#         context = []
#         for i in range(batch_size):
#             st = torch.unsqueeze(states[i, :, :], 0)
#             m1 = torch.bmm(self.att_w1b, torch.transpose(st, 1, 2))
#             m1_actv = self.func_tanh(m1)
#             atts_each_hop = []
#             context_each_hop = []
#             for j in range(self.n_hops):
#                 m2 = torch.bmm(self.att_w2b[:, j, :, :], m1_actv)
#                 m2_actv = self.func_tanh(torch.squeeze(m2))
#                 atts_each_hop.append(m2_actv)
#                 context_each_hop.append(m2_actv * torch.transpose(st, 1, 2))
#             atts_each_hop = torch.stack(atts_each_hop)
#             atts.append(atts_each_hop)
#             context_each_hop = torch.stack(context_each_hop)
#             context.append(context_each_hop)
#         atts = torch.stack(atts)
#         context = torch.stack(context)
#         return atts
#
#
#     # def get_context_vector(self, alpha, beta, vs):
#     #     matrix = beta * vs # element-wise multiplication
#     #     matrix = torch.transpose(matrix, 1, 2)
#     #     alpha = torch.unsqueeze(alpha, 2)
#     #     context = torch.bmm(matrix, alpha)
#     #     context = torch.squeeze(context)
#     #     return context
#     #
#     # def make_step_pred(self, alpha, beta, embedding):
#     #     # linear for context vector to get output at each step
#     #     out = []
#     #     for i in range(self.seq_len):
#     #         context = self.get_context_vector(alpha[:, :i+1], beta[:, :i+1, :], embedding[:, :i+1, :])
#     #         linear_y = self.linear(context)
#     #         out.append(self.func_sigmoid(linear_y))
#     #     out = torch.stack(out)
#     #     out = torch.squeeze(out)
#     #     out = torch.transpose(out, 0, 1)
#     #     return out
#
#     def forward(self, inputs, batch_size):
#         """
#         the recurrent module of the autoencoder
#         """
#         # Embedding
#         embedding_b = self.embedding_layer(inputs)
#         # RNN
#         states_rnn = self.encode_rnn(embedding_b, batch_size)
#         # Add attentions
#         alpha = self.add_attention(states_rnn, batch_size)
#         return alpha, states_rnn
#         # return alpha, beta
#