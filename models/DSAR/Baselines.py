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


class LRmodel(nn.Module):
    def __init__(self, input_dim, output_dim, initrange):
        super(LRmodel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.sigm = nn.Sigmoid()
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        linear = self.linear(inputs)
        output = self.sigm(linear)
        return output, linear


class MLPmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, initrange):
        super(MLPmodel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1, bias=True)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2, bias=True)
        self.linear3 = nn.Linear(hidden_dim2, output_dim, bias=True)
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        linear1 = self.linear1(inputs)
        out1 = self.sigm(linear1)
        linear2 = self.linear2(out1)
        out2 = self.sigm(linear2)
        linear3 = self.linear3(out2)
        output = self.sigm(linear3)
        return output, linear3


class RETAIN(nn.Module):
    """
    A recurrent autoencoder with an encoder, recurrent module, and a decoder.
    In addition, a linear layer is on top of each decode step and the weights are shared at these step.
    RETAIN in PyTorch (Code written by Jinghe Zhang)

    RETAIN: An interpretable predictive model for healthcare using reverse time attention mechanism
    https://arxiv.org/abs/1608.05745
    """

    def __init__(self, input_size, embed_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len, dropout_p=0.5):
        """
        Initilize a recurrent autoencoder
        """
        super(RETAIN, self).__init__()

        # Embedding
        self.embed = nn.Linear(input_size, embed_size, bias=False)
        # Forward RNN
        self.rnn_l = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
                                             batch_first=True, bias=True)
        # Backward RNN
        self.rnn_r = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
                                             batch_first=True, bias=True)
        # linear of attention
        self.linear_att_func1 = nn.Linear(hidden_size, 1, bias=True)
        self.linear_att_func2 = nn.Linear(hidden_size, embed_size, bias=True)
        # final linear layer
        self.linear = nn.Linear(embed_size, output_size, bias=True)

        self.init_weights(initrange)
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
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def init_weights(self, initrange):
        """
        weight initialization
        """
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def embedding_layer(self, inputs):
        # embedding_f = []
        embedding_b = []
        for i in range(self.seq_len):
            embedded = self.embed(inputs[:, :, i])
            # embedding_f.append(embedded)
            embedding_b.insert(0, embedded)
        embedding_b = torch.stack(embedding_b)
        embedding_b = torch.transpose(embedding_b, 0, 1)
        # embedding_b = self.func_tanh(embedding_b)
        return embedding_b

    def encode_rnn_l(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))
        embedding = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn_l(embedding, init_state)
        return outputs_rnn

    def encode_rnn_r(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))
        embedding = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn_r(embedding, init_state)
        return outputs_rnn

    def add_visit_attention(self, states_rnn):
        # attention
        linear_att = []
        for i in range(self.seq_len):
            att = self.linear_att_func1(states_rnn[:, i, :])
            linear_att.append(att)
        linear_att = torch.stack(linear_att)
        linear_att = torch.squeeze(linear_att)
        linear_att = torch.transpose(linear_att, 0, 1)
        alpha = self.func_softmax(linear_att)
        return alpha

    def add_variable_attention(self, states_rnn):
        # attention
        linear_att = []
        for i in range(self.seq_len):
            att = self.linear_att_func2(states_rnn[:, i, :])
            linear_att.append(att)
        linear_att = torch.stack(linear_att)
        linear_att = torch.squeeze(linear_att)
        linear_att = torch.transpose(linear_att, 0, 1)
        beta = self.func_tanh(linear_att)
        return beta

    def get_context_vector(self, alpha, beta, vs):
        matrix = beta * vs # element-wise multiplication
        matrix = torch.transpose(matrix, 1, 2)
        alpha = torch.unsqueeze(alpha, 2)
        context = torch.bmm(matrix, alpha)
        context = torch.squeeze(context)
        return context

    def make_step_pred(self, alpha, beta, embedding):
        # linear for context vector to get output at each step
        out = []
        for i in range(self.seq_len):
            context = self.get_context_vector(alpha[:, :i+1], beta[:, :i+1, :], embedding[:, :i+1, :])
            linear_y = self.linear(context)
            out.append(self.func_sigmoid(linear_y))
        out = torch.stack(out)
        out = torch.squeeze(out)
        out = torch.transpose(out, 0, 1)
        return out

    def forward(self, inputs, batch_size):
        """
        the recurrent module of the autoencoder
        """
        # Embedding
        embedding_b = self.embedding_layer(inputs)
        # RNN on the left for variable level attention, backward RNN
        states_rnn_l = self.encode_rnn_l(embedding_b, batch_size)
        # RNN on the right for visit level attention, backward RNN
        states_rnn_r = self.encode_rnn_r(embedding_b, batch_size)
        # Add attentions
        alpha = self.add_visit_attention(states_rnn_r)
        beta = self.add_variable_attention(states_rnn_l)
        # linear for context vector to get final output
        # context = self.get_context_vector(alpha, beta, embedding_b)
        # linear_y = self.linear(context)
        # out = self.func_softmax(linear_y)
        out = self.make_step_pred(alpha, beta, embedding_b)
        return out, [alpha, beta]
        # return alpha, beta
