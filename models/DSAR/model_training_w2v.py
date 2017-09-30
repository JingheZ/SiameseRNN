"""
To train the RNN models
"""

import os
import pickle
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from models.DSAR.Patient2Vec import Patient2Vec, Patient2Vec0
from torch.autograd import Variable
from gensim.models import Word2Vec
# from models.DSAR.Baselines import RNNmodel
import numpy as np
from sklearn import metrics


class RNNmodel(nn.Module):
    """
    A recurrent NN
    """
    def __init__(self, input_size, embed_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len, bi, ct, dropout_p=0.5):
        """
        Initilize a recurrent autoencoder
        """
        super(RNNmodel, self).__init__()

        # Embedding
        self.embed = nn.Linear(input_size, embed_size, bias=False)
        # RNN
        self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout_p,
                                             batch_first=True, bias=True, bidirectional=bi)
        self.b = 1
        if bi:
            self.b = 2
        self.linear = nn.Linear(hidden_size * self.b, output_size, bias=True)
        self.tanh = nn.Hardtanh()
        self.init_weights(initrange)
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.ct = ct
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

    def embedding_layer(self, inputs, inputs_demoips):
        if self.ct:
            inputs_agg = inputs
        else:
            inputs_agg = torch.sum(inputs, dim=2)
            inputs_agg = torch.squeeze(inputs_agg, dim=2)
        embedding = []
        for i in range(self.seq_len):
            embedded = self.embed(torch.cat((inputs_agg[:, i], inputs_demoips), 1))
            embedding.append(embedded)
        embedding = torch.stack(embedding)
        embedding = torch.transpose(embedding, 0, 1)
            # embedding = self.tanh(embedding)
        return embedding

    def encode_rnn(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers * self.b, batch_size, self.hidden_size).zero_()))
        # embedding_d = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn(embedding, init_state)
        return outputs_rnn

    def forward(self, inputs, inputs_demoips, batch_size):
        """
        the recurrent module
        """
        # Embedding
        embedding = self.embedding_layer(inputs, inputs_demoips)
        # embedding = torch.transpose(inputs, 1, 2)
        # RNN
        states_rnn = self.encode_rnn(embedding, batch_size)
        # linear for context vector to get final output
        linear_y = self.linear(states_rnn[:, -1])
        out = self.func_softmax(linear_y)
        # out = self.func_sigmoid(linear_y)
        # out = self.func_tanh(linear_y)
        return out, [states_rnn, embedding, linear_y]


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
        self.att_w1 = nn.Linear(hidden_size, att_dim, bias=False)

        # final linear layer
        self.linear = nn.Linear(hidden_size + 3, output_size, bias=True)

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
        init_state = (Variable(self.weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))
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

    def forward(self, inputs, inputs_demoip, batch_size):
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
        # Final linear layer with demographic and previous IP info added as extra variables
        context_v2 = torch.cat((context, inputs_demoip), 1)
        linear_y = self.linear(context_v2)
        out = self.func_softmax(linear_y)
        return out, [states_rnn, context, alpha]


def create_batch(step, batch_size, data_x, data_demoip, data_y, w2v, vsize, pad_size):
    start = step * batch_size
    end = (step + 1) * batch_size
    batch_x = []
    for i in range(start, end):
        x = create_sequence(data_x[i], w2v, vsize, pad_size)
        batch_x.append(x)
    batch_demoip = data_demoip[start:end]
    batch_y = data_y[start:end]
    return Variable(torch.FloatTensor(batch_x), requires_grad=False), \
           Variable(torch.FloatTensor(batch_demoip), requires_grad=False),\
           Variable(torch.LongTensor(batch_y), requires_grad=False) # for cross-entropy loss
    # return Variable(torch.FloatTensor(batch_x), requires_grad=False), Variable(torch.FloatTensor(batch_y), requires_grad=False) # for cross-entropy loss


def create_full_set(dt, y, w2v, vsize, pad_size):
    x = []
    for i in range(len(dt)):
        seq = create_sequence(dt[i], w2v, vsize, pad_size)
        x.append(seq)
    return x, y


def create_sequence(items, w2v, dim, pad_size):
    seq = [[], [], [], [], [], [], [], [], [], [], [], []]
    seq_flag = [0] * 12
    for l in range(12):
        seq_flag[l] = len(items[l])
        if seq_flag[l] > 0:
            for j in range(seq_flag[l]):
                vec = w2v[items[l][j]].tolist()
                seq[l].append(vec)
        # pad the input events in a visit to pad_size
        if seq_flag[l] < pad_size:
            seq[l] += [[0] * dim for k in range(pad_size - seq_flag[l])]
    return seq


def list2tensor(x, y):
    x = Variable(torch.FloatTensor(x), requires_grad=False)
    y = Variable(torch.LongTensor(y), requires_grad=False)
    # y = Variable(torch.FloatTensor(y), requires_grad=False)
    return x, y


def tensor2scalor(mat):
    return mat.view(-1).data.tolist()[0]


def process_demoip():
    with open('./data/hospitalization_train_data_demoip.pickle', 'rb') as f:
        train_genders, train_ages, train_ip = pickle.load(f)
    f.close()

    with open('./data/hospitalization_validate_data_demoip.pickle', 'rb') as f:
        validate_genders, validate_ages, validate_ip = pickle.load(f)
    f.close()

    with open('./data/hospitalization_test_data_demoip.pickle', 'rb') as f:
        test_genders, test_ages, test_ip = pickle.load(f)
    f.close()
    train = np.vstack((train_genders, train_ages, train_ip)).transpose().tolist()
    validate = np.vstack((validate_genders, validate_ages, validate_ip)).transpose().tolist()
    test = np.vstack((test_genders, test_ages, test_ip)).transpose().tolist()
    return train, validate, test


def model_testing_one_batch(model, batch_x, batch_demoip, batch_size):
    y_pred, _ = model(batch_x, batch_demoip, batch_size)
    _, predicted = torch.max(y_pred.data, 1)
    pred = predicted.view(-1).tolist()
    return pred


def model_testing(model, test, test_y, test_demoips, w2v, vsize, pad_size, batch_size=1000):
    i = 0
    pred_all = []
    while (i + 1) * batch_size <= len(test_y):
        batch_x, batch_demoip, _ = create_batch(i, batch_size, test, test_demoips, test_y, w2v, vsize, pad_size)
        pred = model_testing_one_batch(model, batch_x, batch_demoip, batch_size)
        pred_all += pred
        i += 1
    # the remaining data less than one batch
    batch_demoip = test_demoips[i * batch_size:]
    batch_x = []
    for j in range(i * batch_size, len(test_y)):
        x = create_sequence(test[j], w2v, vsize, pad_size)
        batch_x.append(x)
    batch_x = Variable(torch.FloatTensor(batch_x), requires_grad=False)
    batch_demoip = Variable(torch.FloatTensor(batch_demoip), requires_grad=False)
    pred = model_testing_one_batch(model, batch_x, batch_demoip, len(test_y) - i * batch_size)
    pred_all += pred
    return pred_all


def calculate_performance(test_y, pred):
    # calculate performance
    perfm = metrics.classification_report(test_y, pred)
    auc = metrics.roc_auc_score(test_y, pred)
    return perfm, auc


if __name__ == '__main__':

    #  ============== Prepare Data ===========================
    # get demographic and previous IP info
    train_demoips, validate_demoips, test_demoips = process_demoip()

    # model_type = 'rnn-rt'
    # model_type = 'retain'
    # model_type = 'rnn-bi'
    # model_type = 'rnn-rt-bi'
    # model_type = 'patient2vec'
    # ----- load word2vec embedding model
    size = 100
    window = 100
    sg = 1 # skip-gram:1; cbow: 0
    model_path = './results/w2v_size' + str(size) + '_window' + str(window) + '_sg' + str(sg)
    w2v_model = Word2Vec.load(model_path)

    with open('./data/hospitalization_train_validate_test_ids.pickle', 'rb') as f:
        train_ids, valid_ids, test_ids = pickle.load(f)
    f.close()

    with open('./data/hospitalization_train_data.pickle', 'rb') as f:
        train, train_y = pickle.load(f)
    f.close()

    with open('./data/hospitalization_validate_data.pickle', 'rb') as f:
        validate, validate_y = pickle.load(f)
    f.close()

    with open('./data/hospitalization_test_data.pickle', 'rb') as f:
        test, test_y = pickle.load(f)
    f.close()


    # create input tensor and pad each visit to length of 200
    pad_size = 76
    size = 100
    # Prepare validation data for the model
    validate_x, validate_y = create_full_set(validate, validate_y, w2v_model, size, pad_size)
    # with open('./data/hospitalization_validate_data_padded.pickle', 'wb') as f:
    #     pickle.dump([validate_x, validate_y], f)
    # f.close()
    validate_x, validate_y = list2tensor(validate_x, validate_y)
    validate_demoips = Variable(torch.FloatTensor(validate_demoips), requires_grad=False)
    # Model hyperparameters
    # model_type = 'rnn-rt'
    input_size = size + 3
    embedding_size = 100
    hidden_size = 256
    n_layers = 1
    seq_len = 12
    output_size = 2
    rnn_type = 'GRU'
    drop = 0.0
    learning_rate = 0.001
    decay = 0.005
    interval = 100
    initrange = 1
    att_dim = 1
    n_hops = 5
    batch_size = 100
    epoch_max = 10 # training for maximum 3 epochs of training data
    n_iter_max_dev = 1000 # if no improvement on dev set for maximum n_iter_max_dev, terminate training
    train_iters = len(train_ids)

    model_type = 'crnn'
    # Build and train/load the model
    print('Build Model...')
    # by default build a LR model
    if model_type == 'rnn':
        model = RNNmodel(input_size, embedding_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len,
                         ct=False, bi=False, dropout_p=drop)
    elif model_type == 'rnn-bi':
        model = RNNmodel(input_size, embedding_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len,
                         ct=False, bi=True, dropout_p=drop)
    # elif model_type == 'patient2vec':
    #     model = Patient2Vec(input_size, embedding_size, hidden_size, n_layers, n_hops, att_dim, initrange, output_size,
    #                         rnn_type, seq_len, pad_size, dropout_p=drop)
    elif model_type == 'crnn':
        model = Patient2Vec0(input_size - 3, embedding_size, hidden_size, n_layers, att_dim, initrange, output_size,
                            rnn_type, seq_len, pad_size, dropout_p=drop)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 11]))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    model_path = './saved_models/model_' + model_type + '_layer' + str(n_layers) + '.dat'
    print('Start Training...')
    # if os.path.exists(model_path):
    #     saved_model = torch.load(model_path)
    #     model.load_state_dict(saved_model)
    # # else:
        # model.init_weights(initrange)
        # Train the model
    start_time = time.time()
    best_loss_dev = 100
    best_dev_iter = 0
    n_iter = 0
    epoch = 0
    while epoch < epoch_max:
        step = 0
        while (step + 1) * batch_size < train_iters:
            batch_x, batch_demoip, batch_y = create_batch(step, batch_size, train, train_demoips, train_y, w2v_model, size, pad_size)
            optimizer.zero_grad()
            y_pred, _ = model(batch_x, batch_demoip, batch_size)
            # states, alpha, beta = model(batch_x, batch_size)
            loss = criterion(y_pred, batch_y)
            # loss = CrossEntropy_Multi(y_pred, batch_y, output_size, criterion)
            # loss = get_loss(y_pred, batch_y, criterion, seq_len)
            loss.backward()
            optimizer.step()

            if step % interval == 0:
                elapsed = time.time() - start_time
                # acc = calcualte_accuracy(y_pred, batch_y, batch_size)
                print('%i epoch, %i batches, elapsed time: %.2f, loss: %.3f' % (epoch + 1, step + 1, elapsed, loss.data[0]))
                # Evaluate model performance on validation set
                pred_dev, _ = model(validate_x, validate_demoips, len(valid_ids))
                loss_dev = criterion(pred_dev, validate_y)
                pred_ind_dev = model_testing_one_batch(model, validate_x, validate_demoips,
                                                       len(valid_ids))
                perfm_dev, auc_dev = calculate_performance(validate_y.data.tolist(), pred_ind_dev)
                print("Performance on dev set: AUC is %.3f" % auc_dev)
                # print(perfm_dev)

                pred_ind_batch = model_testing_one_batch(model, batch_x, batch_demoip, batch_size)
                perfm_batch, auc_batch = calculate_performance(batch_y.data.tolist(), pred_ind_batch)
                print("Performance on training set: AUC is %.3f" % auc_batch)
                # print(perfm_batch)
                print('Validation, loss: %.3f' % (loss_dev.data[0]))
                # if loss_dev < best_loss_dev:
                #     best_loss_dev = loss_dev
                #     best_dev_iter = n_iter
                # if n_iter - best_dev_iter >= n_iter_max_dev:
                #     break
            step += 1
            # n_iter += 1
        # if n_iter - best_dev_iter >= n_iter_max_dev:
        #     break
        epoch += 1
    # save trained model
    state_to_save = model.state_dict()
    torch.save(state_to_save, model_path)
    elapsed = time.time() - start_time
    print('Training Finished! Total Training Time is: % .2f' % elapsed)
    #
    # # ============================ To evaluate model using testing set =============================================
    print('Start Testing...')
    result_file = './results/test_results_' + model_type + '_layer' + str(n_layers) + '.pickle'
    # output_file = './results/test_outputs_' + model_type + '_layer' + str(n_layers) + '.pickle'

    # # Evaluate the model
    model.eval()
    test_start_time = time.time()
    pred_test = model_testing(model, test, test_y, test_demoips, w2v_model, size, pad_size, batch_size=1000)
    perfm, auc = calculate_performance(test_y, pred_test)
    elapsed_test = time.time() - test_start_time
    print(auc)
    print(perfm)
    with open(result_file, 'wb') as f:
        pickle.dump([pred_test, test_y], f)
    f.close()
    print('Testing Finished!')
    # with open(output_file, 'wb') as f:
    #     pickle.dump(output_test, f)
    # f.close()
    # print(pred_test[:10, :10])
    # # To do:
    # # 1. When selecting variables, for labs select last or first two and last two;
    # # while for meds and dxs, code as 1 when the item is ever true in the patient data
    #
    # # Prepare test data for the model
    # test_x, test_y = create_full_set(test, test_y, w2v_model, size, pad_size)
    # with open('./data/hospitalization_test_data_padded.pickle', 'wb') as f:
    #     pickle.dump([test_x, test_y], f)
    # f.close()
    # test_x, test_y = list2tensor(test_x, test_y)