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
from models.DSAR.Patient2Vec import Patient2Vec, Patient2Vec0
from torch.autograd import Variable
from gensim.models import Word2Vec
from models.DSAR.Baselines import RNNmodel
import numpy as np
from sklearn import metrics


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
    for i in range(i * batch_size, len(test_y)):
        x = create_sequence(test[i], w2v, vsize, pad_size)
        batch_x.append(x)
    batch_x = Variable(torch.FloatTensor(batch_x), requires_grad=False)
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

    # model_type = 'rnn'
    # model_type = 'rnn-rt'
    # model_type = 'retain'
    # model_type = 'rnn-bi'
    # model_type = 'rnn-rt-bi'
    model_type = 'patient2vec'
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
    pad_size = 115
    size = 100
    # Prepare validation data for the model
    validate_x, validate_y = create_full_set(validate, validate_y, w2v_model, size, pad_size)
    with open('./data/hospitalization_validate_data_padded.pickle', 'wb') as f:
        pickle.dump([validate_x, validate_y], f)
    f.close()
    validate_x, validate_y = list2tensor(validate_x, validate_y)
    validate_demoips = Variable(torch.FloatTensor(validate_demoips), requires_grad=False)
    # Model hyperparameters
    # model_type = 'rnn-rt'
    input_size = size + 3
    embedding_size = 150
    hidden_size = 64
    n_layers = 1
    seq_len = 12
    output_size = 2
    rnn_type = 'GRU'
    drop = 0.0
    learning_rate = 0.001
    decay = 0.01
    interval = 10
    initrange = 1
    att_dim = 1
    n_hops = 5
    batch_size = 100
    epoch_max = 30 # training for maximum 3 epochs of training data
    n_iter_max_dev = 1000 # if no improvement on dev set for maximum n_iter_max_dev, terminate training
    train_iters = len(train_ids)

    # Build and train/load the model
    print('Build Model...')
    # by default build a LR model
    if model_type == 'rnn':
        model = RNNmodel(input_size, embedding_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len,
                         ct=False, bi=False, dropout_p=drop)
    elif model_type == 'rnn-bi':
        model = RNNmodel(input_size, embedding_size, hidden_size, n_layers, initrange, output_size, rnn_type, seq_len,
                         ct=False, bi=True, dropout_p=drop)
    elif model_type == 'patient2vec':
        model = Patient2Vec(input_size, embedding_size, hidden_size, n_layers, n_hops, att_dim, initrange, output_size,
                            rnn_type, seq_len, pad_size, dropout_p=drop)
    elif model_type == 'patient2vec':
        model = Patient2Vec0(input_size, embedding_size, hidden_size, n_layers, att_dim, initrange, output_size,
                            rnn_type, seq_len, pad_size, dropout_p=drop)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    model_path = './saved_models/model_' + model_type + '_layer' + str(n_layers) + '.dat'
    print('Start Training...')
    if os.path.exists(model_path):
        saved_model = torch.load(model_path)
        model.load_state_dict(saved_model)
    else:
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
                    print(perfm_dev)

                    pred_ind_batch = model_testing_one_batch(model, batch_x, batch_demoip, batch_size)
                    perfm_batch, auc_batch = calculate_performance(batch_y.data.tolist(), pred_ind_batch)
                    print("Performance on training set: AUC is %.3f" % auc_batch)
                    print(perfm_batch)
                    print('Validation, loss: %.3f' % (loss_dev.data[0]))
                    # if loss_dev < best_loss_dev:
                    #     best_loss_dev = loss_dev
                    #     best_dev_iter = n_iter
                    # if n_iter - best_dev_iter >= n_iter_max_dev:
                    #     break
                step += 1
                n_iter += 1
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
    output_file = './results/test_outputs_' + model_type + '_layer' + str(n_layers) + '.pickle'

    # # Evaluate the model
    model.eval()
    test_start_time = time.time()
    pred_test = model_testing(model, test, test_y, test_demoips, w2v_model, size, pad_size, batch_size=1000)
    perfm, auc = calculate_performance(test_y, pred_test)
    elapsed_test = time.time() - test_start_time
    print(auc)
    print(perfm)
    # with open(result_file, 'wb') as f:
    #     pickle.dump([pred_test, test_y], f)
    # f.close()
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