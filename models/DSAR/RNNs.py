"""
A conventional GRU model in PyTorch (Code written by Jinghe Zhang)
the first one RNNmodel with prediction at the end
the second model RNNmodelRT with prediction at each time step
both DoctorAI and Learning2Diagnose are originated from these model
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNmodel(nn.Module):
    """
    A recurrent NN
    """

    def __init__(self, input_size, embed_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len, bi, dropout_p=0.5):
        """
        Initilize a recurrent autoencoder
        """
        super(RNNmodel, self).__init__()

        # Embedding
        self.embed = nn.Linear(input_size, embed_size, bias=False)
        # RNN
        self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
                                             batch_first=True, bias=True, bidirectional=bi)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.tanh = nn.Hardtanh()
        self.init_weights(initrange)
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.func_softmax = nn.Softmax()
        self.func_sigmoid = nn.Sigmoid()
        self.func_tanh = nn.Hardtanh(0, 1)
        # Add dropout
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

    def init_weights(self, initrange=1):
        """
        weight initialization
        """
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)
            # param.data.normal_(0, 1)

    def embedding_layer(self, inputs):
        inputs_agg = torch.sum(inputs, dim=2)
        inputs_agg = torch.squeeze(inputs_agg, dim=2)
        embedding = []
        for i in range(self.seq_len):
            embedded = self.embed(inputs_agg[:, i])
            embedding.append(embedded)
        embedding = torch.stack(embedding)
        embedding = torch.transpose(embedding, 0, 1)
        # embedding = self.tanh(embedding)
        return embedding

    def encode_rnn(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))
        # embedding_d = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn(embedding, init_state)
        return outputs_rnn

    def forward(self, inputs, batch_size):
        """
        the recurrent module
        """
        # Embedding
        embedding = self.embedding_layer(inputs)
        # embedding = torch.transpose(inputs, 1, 2)
        # RNN
        states_rnn = self.encode_rnn(embedding, batch_size)
        # linear for context vector to get final output
        linear_y = self.linear(states_rnn[:, -1])
        out = self.func_softmax(linear_y)
        # out = self.func_sigmoid(linear_y)
        # out = self.func_tanh(linear_y)
        return out, [states_rnn, embedding, linear_y]
#
#
# class RNNmodelRT(nn.Module):
#     """
#     A recurrent NN with replicate targets
#     """
#
#     def __init__(self, input_size, embed_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len, dropout_p=0.5):
#         """
#         Initilize a recurrent autoencoder
#         """
#         super(RNNmodelRT, self).__init__()
#
#         # Embedding
#         self.embed = nn.Linear(input_size, embed_size, bias=False)
#         # RNN
#         self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
#                                              batch_first=True, bias=True)
#         self.linear = nn.Linear(hidden_size, output_size, bias=True)
#         self.tanh = nn.Hardtanh()
#         self.init_weights(initrange)
#         self.input_size = input_size
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.seq_len = seq_len
#         self.func_sigmoid = nn.Sigmoid()
#         self.func_tanh = nn.Hardtanh(0, 1)
#         # Add dropout
#         self.dropout_p = dropout_p
#         self.dropout = nn.Dropout(p=self.dropout_p)
#
#     def init_weights(self, initrange):
#         """
#         weight initialization
#         """
#         for param in self.parameters():
#             param.data.uniform_(-initrange, initrange)
#
#     def embedding_layer(self, inputs):
#         embedding_f = []
#         for i in range(self.seq_len):
#             embedded = self.embed(inputs[:, :, i])
#             embedding_f.append(embedded)
#         embedding_f = torch.stack(embedding_f)
#         embedding_f = torch.transpose(embedding_f, 0, 1)
#         embedding_f = self.tanh(embedding_f)
#         return embedding_f
#
#     def encode_rnn(self, embedding, batch_size):
#         self.weight = next(self.parameters()).data
#         init_state = (Variable(self.weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))
#         # embedding = self.dropout(embedding)
#         outputs_rnn, states_rnn = self.rnn(embedding, init_state)
#         return outputs_rnn
#
#     def make_step_pred(self, states):
#         # linear for context vector to get output at each step
#         out = []
#         for i in range(self.seq_len):
#             linear_y = self.linear(states[:, i, :])
#             out.append(self.func_sigmoid(linear_y))
#             # out.append(self.func_tanh(linear_y))
#         out = torch.stack(out)
#         out = torch.squeeze(out)
#         out = torch.transpose(out, 0, 1)
#         return out
#
#     def forward(self, inputs, batch_size):
#         """
#         the recurrent module
#         """
#         # Embedding
#         embedding = self.embedding_layer(inputs)
#         # embedding = torch.transpose(inputs, 1, 2)
#         # RNN
#         states_rnn = self.encode_rnn(embedding, batch_size)
#         # final output
#         out = self.make_step_pred(states_rnn)
#         return out, states_rnn


# class RNNmodelRTBi(nn.Module):
#     """
#     A recurrent NN
#     """
#
#     def __init__(self, input_size, embed_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len, dropout_p=0.5):
#         """
#         Initilize a recurrent autoencoder
#         """
#         super(RNNmodelRTBi, self).__init__()
#
#         # Embedding
#         self.embed = nn.Linear(input_size, embed_size, bias=False)
#         # RNN
#         self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
#                                              batch_first=True, bias=True, bidirectional=True)
#         self.linear = nn.Linear(hidden_size * 2, output_size, bias=True)
#         self.tanh = nn.Hardtanh()
#         self.init_weights(initrange)
#         self.input_size = input_size
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.seq_len = seq_len
#         self.func_sigmoid = nn.Sigmoid()
#         self.func_tanh = nn.Hardtanh(0, 1)
#         # Add dropout
#         self.dropout_p = dropout_p
#         self.dropout = nn.Dropout(p=self.dropout_p)
#
#     def init_weights(self, initrange=0.1):
#         """
#         weight initialization
#         """
#         for param in self.parameters():
#             param.data.uniform_(-initrange, initrange)
#
#     def embedding_layer(self, inputs):
#         embedding_f = []
#         for i in range(self.seq_len):
#             embedded = self.embed(inputs[:, :, i])
#             embedding_f.append(embedded)
#         embedding_f = torch.stack(embedding_f)
#         embedding_f = torch.transpose(embedding_f, 0, 1)
#         embedding_f = self.tanh(embedding_f)
#         return embedding_f
#
#     def encode_rnn(self, embedding, batch_size):
#         self.weight = next(self.parameters()).data
#         init_state = (Variable(self.weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_()))
#         # embedding = self.dropout(embedding)
#         outputs_rnn, states_rnn = self.rnn(embedding, init_state)
#         return outputs_rnn
#
#     def make_step_pred(self, states):
#         # linear for context vector to get output at each step
#         out = []
#         for i in range(self.seq_len):
#             linear_y = self.linear(states[:, i, :])
#             out.append(self.func_sigmoid(linear_y))
#             # out.append(self.func_tanh(linear_y))
#         out = torch.stack(out)
#         out = torch.squeeze(out)
#         out = torch.transpose(out, 0, 1)
#         return out
#
#     def forward(self, inputs, batch_size):
#         """
#         the recurrent module
#         """
#         # Embedding
#         embedding = self.embedding_layer(inputs)
#         # embedding = torch.transpose(inputs, 1, 2)
#         # RNN
#         states_rnn = self.encode_rnn(embedding, batch_size)
#         # final output
#         out = self.make_step_pred(states_rnn)
#         return out, states_rnn
