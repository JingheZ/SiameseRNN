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
# from models.Patient2Vec import Patient2Vec
from torch.autograd import Variable
from gensim.models import Word2Vec
# from models.DSAR.Baselines import RNNmodel, RETAIN
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

    def embedding_layer(self, inputs, inputs_demoips):
        # embedding_f = []
        embedding_b = []
        for i in range(self.seq_len):
            # embedded = self.embed(inputs[:, :, i])
            embedded = self.embed(torch.cat((inputs[:, i], inputs_demoips), 1))
            # embedding_f.append(embedded)
            embedding_b.insert(0, embedded)
        embedding_b = torch.stack(embedding_b)
        embedding_b = torch.transpose(embedding_b, 0, 1)
        # embedding_b = self.func_tanh(embedding_b)
        return embedding_b

    def encode_rnn_l(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))
        # embedding = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn_l(embedding, init_state)
        return outputs_rnn

    def encode_rnn_r(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))
        # embedding = self.dropout(embedding)
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

    def forward(self, inputs, inputs_demoips, batch_size):
        """
        the recurrent module of the autoencoder
        """
        # Embedding
        embedding_b = self.embedding_layer(inputs, inputs_demoips)
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


def create_batch(step, batch_size, data_x, data_demoip, data_y):
    start = step * batch_size
    end = (step + 1) * batch_size
    batch_x = data_x[start:end]
    batch_demoip = data_demoip[start:end]
    batch_y = data_y[start:end]
    batch_x, batch_y = list2tensor(batch_x, batch_y)
    return batch_x, Variable(torch.FloatTensor(batch_demoip), requires_grad=False), batch_y # for cross-entropy loss
    # return Variable(torch.FloatTensor(batch_x), requires_grad=False), Variable(torch.FloatTensor(batch_y), requires_grad=False) # for cross-entropy loss


def list2tensor(x, y):
    x = Variable(torch.FloatTensor(x), requires_grad=False)
    y = Variable(torch.LongTensor(y), requires_grad=False)
    # y = Variable(torch.FloatTensor(y), requires_grad=False)
    x = torch.split(x, split_size=12, dim=1)
    x = torch.stack(x, dim=1)
    x = torch.transpose(x, 1, 2)
    return x, y


def tensor2scalor(mat):
    return mat.view(-1).data.tolist()[0]


def get_loss(pred, y, criterion, seq_len):
    loss = Variable(torch.FloatTensor([0]))
    for t in range(seq_len):
        loss = torch.add(loss, criterion(pred[:, t, :], y))
    loss = torch.div(loss, seq_len)
    return loss


def get_top_dxs_inds(response, n=10):
    filename = './data/dxs_counts.csv'
    data = pd.read_csv(filename)
    dxs = data['dx'][:n].values
    inds = [response.index(x) for x in dxs]
    return inds


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


def model_testing_one_batch(model, model_type, batch_x, batch_demoip, batch_size):
    y_pred, _ = model(batch_x, batch_demoip, batch_size)
    if model_type == 'retain':
        _, predicted = torch.max(y_pred.data[:, -1, :], 1)
    else:
        _, predicted = torch.max(y_pred.data, 1)
    pred = predicted.view(-1).tolist()
    return pred


def model_testing(model, model_type, test, test_y, test_demoips, batch_size=1000):
    i = 0
    pred_all = []
    while (i + 1) * batch_size <= len(test_y):
        batch_x, batch_demoip, _ = create_batch(i, batch_size, test, test_demoips, test_y)
        pred = model_testing_one_batch(model, model_type, batch_x, batch_demoip, batch_size)
        pred_all += pred
        i += 1
    # the remaining data less than one batch
    batch_x = test[i * batch_size:]
    batch_x = Variable(torch.FloatTensor(batch_x), requires_grad=False)
    batch_x = torch.split(batch_x, split_size=12, dim=1)
    batch_x = torch.stack(batch_x, dim=1)
    batch_x = torch.transpose(batch_x, 1, 2)
    batch_demoip = test_demoips[i * batch_size:]
    batch_demoip = Variable(torch.FloatTensor(batch_demoip), requires_grad=False)
    pred = model_testing_one_batch(model, model_type, batch_x, batch_demoip, len(test_y) - i * batch_size)
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

    with open('./data/hospitalization_train_validate_test_ids.pickle', 'rb') as f:
        train_ids, valid_ids, test_ids = pickle.load(f)
    f.close()

    with open('./data/hospitalization_train_data_sub_cts.pickle', 'rb') as f:
        train, train_y = pickle.load(f)
    f.close()
    #
    with open('./data/hospitalization_validate_data_sub_cts.pickle', 'rb') as f:
        validate, validate_y = pickle.load(f)
    f.close()
    #
    with open('./data/hospitalization_test_data_sub_cts.pickle', 'rb') as f:
        test, test_y = pickle.load(f)
    f.close()

    # # Prepare validation data for the model
    validate_x, validate_y = list2tensor(validate, validate_y)
    validate_demoips = Variable(torch.FloatTensor(validate_demoips), requires_grad=False)

    with open('./data/hospitalization_cts_sub_columns.pickle', 'rb') as f:
        features = pickle.load(f)
    f.close()
    # Model hyperparameters
    # model_type = 'rnn-rt'
    input_size = int(len(features)/12) + 3
    embedding_size = 300
    hidden_size = 256
    n_layers = 1
    seq_len = 12
    output_size = 2
    rnn_type = 'GRU'
    drop = 0.0
    learning_rate = 0.0005
    decay = 0.005
    interval = 100
    initrange = 1
    att_dim = 100

    batch_size = 100
    epoch_max = 3 # training for maximum 3 epochs of training data
    n_iter_max_dev = 1000 # if no improvement on dev set for maximum n_iter_max_dev, terminate training
    train_iters = len(train_ids)
    model_type = 'rnn-bi'
    # Build and train/load the model
    print('Build Model...')
    # by default build a RNN model
    model = RNNmodel(input_size, embedding_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len,
                     bi=False, ct=True, dropout_p=drop)
    if model_type == 'rnn-bi':
        model = RNNmodel(input_size, embedding_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len,
                         bi=True, ct=True, dropout_p=drop)
    elif model_type == 'retain':
        model = RETAIN(input_size, embedding_size, hidden_size, n_layers, initrange, output_size,
                            rnn_type, seq_len, dropout_p=drop)
    # criterion = nn.MultiLabelMarginLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 20]))
    # criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    model_path = './saved_models/model_' + model_type + '_layer' + str(n_layers) + '.dat'
    print('Start Training...')
    # if os.path.exists(model_path):
    #     saved_model = torch.load(model_path)
    #     model.load_state_dict(saved_model)
    # else:
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
            batch_x, batch_demoip, batch_y = create_batch(step, batch_size, train, train_demoips, train_y)
            optimizer.zero_grad()
            y_pred, _ = model(batch_x, batch_demoip, batch_size)
            # states, alpha, beta = model(batch_x, batch_size)
            if model_type == 'retain':
                loss = get_loss(y_pred, batch_y, criterion, seq_len)
            else:
                loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            if step % interval == 0:
                elapsed = time.time() - start_time
                # acc = calcualte_accuracy(y_pred, batch_y, batch_size)
                print('%i epoch, %i batches, elapsed time: %.2f, loss: %.3f' % (epoch + 1, step + 1, elapsed, loss.data[0]))
                # Evaluate model performance on validation set
                pred_dev, _ = model(validate_x, validate_demoips, len(valid_ids))
                if model_type == 'retain':
                    loss_dev = get_loss(pred_dev, validate_y, criterion, seq_len)
                else:
                    loss_dev = criterion(pred_dev, validate_y)
                pred_ind_dev = model_testing_one_batch(model, model_type, validate_x, validate_demoips,
                                                                len(valid_ids))
                perfm_dev, auc_dev = calculate_performance(validate_y.data.tolist(), pred_ind_dev)
                print("Performance on dev set: AUC is %.3f" % auc_dev)
                # print(perfm_dev)

                pred_ind_batch = model_testing_one_batch(model, model_type, batch_x, batch_demoip, batch_size)
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

    # ============================ To evaluate model using testing set =============================================
    print('Start Testing...')
    result_file = './results/test_results_' + model_type + '_layer' + str(n_layers) + '.pickle'
    # output_file = './results/test_outputs_' + model_type + '_layer' + str(n_layers) + '.pickle'

    # Evaluate the model
    model.eval()
    test_start_time = time.time()
    pred_test = model_testing(model, model_type, test, test_y, test_demoips, batch_size=1000)
    perfm, auc = calculate_performance(test_y, pred_test)
    elapsed_test = time.time() - test_start_time
    print(auc)
    print(perfm)
    with open(result_file, 'wb') as f:
        pickle.dump([pred_test, test_y], f)
    f.close()
    # print('Testing Finished!')
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
