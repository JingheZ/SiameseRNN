"""
Prepare data for experiments
"""

import pickle
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import time


def split_train_validate_test(pos, neg):
    ids = np.array(list(pos) + list(neg))
    ys = [1] * len(pos) + [0] * len(neg)
    rs = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.15, random_state=1)
    train_ids = []
    test_ids = []
    valid_ids = []
    for train_ind, test_ind in rs.split(ids, ys):
        train_ids, test_ind = ids[train_ind], ids[test_ind]
        valid_ids = set(ids).difference(set(train_ids).union(set(test_ind)))
    return train_ids, test_ids, valid_ids


def find_previous_IP(dt_pos, dt_neg):
    pass


def group_items_byadmmonth(dt):
    dt_window = dt[['ptid', 'itemid', 'adm_month']].groupby(['ptid', 'adm_month'])['itemid'].apply(list)
    dt_window = dt_window.reset_index()



if __name__ == '__main__':

    # =============== learn item embedding =================================
    with open('./data/clinical_events_hospitalization.pickle', 'rb') as f:
        data = pickle.load(f)
    f.close()
    data = data[['vid', 'itemid']].drop_duplicates()
    docs = data.groupby('vid')['itemid'].apply(list)
    docs = docs.reset_index()
    docs['length'] = docs['itemid'].apply(lambda x: len(x))
    docs['length'].describe() # 80% quantile is 10; 92.5% quantile is 20; 98.75% quantile is 100
    # all itemids by visit
    vts = docs['itemid'].values.tolist()
    # run the skip-gram w2v model
    size = 100
    window = 100
    min_count = 50
    workers = 28
    iter = 5
    sg = 1 # skip-gram:1; cbow: 0
    model_path = './results/w2v_size' + str(size) + '_window' + str(window) + '_sg' + str(sg)
    # if os.path.exists(model_path):
    #     model = Word2Vec.load(model_path)
    # else:
    a = time.time()
    model = Word2Vec(docs, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter)
    model.save(model_path)
    b = time.time()
    print('training time (mins): %.3f' % ((b - a) / 60)) # 191 mins
    # vocab = list(model.wv.vocab.keys())
    # c = vocab[1]
    # sims = model.most_similar(c)
    # print(c)
    # print(sims)
    #
    #
    #
    #
    #
    # with open('./data/hospitalization_data_pos_neg_ids.pickle', 'rb') as f:
    #     pos_ids, neg_ids = pickle.load(f)
    # f.close()
    #
    # with open('./data/hospitalization_data_1year.pickle', 'rb') as f:
    #     dt = pickle.load(f)
    # f.close()
    #
    # dt_pos = dt[dt['ptid'].isin(pos_ids)]
    # dt_neg = dt[dt['ptid'].isin(neg_ids)]
    # dt_pos['y'] = 1
    # dt_neg['y'] = 1
    # train_ids, valid_ids, test_ids = split_train_validate_test(pos_ids, neg_ids)
    #
