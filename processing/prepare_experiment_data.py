"""
Prepare data for experiments
"""

import pickle
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
import pandas as pd


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

    with open('./data/hospitalization_data_pos_neg_ids.pickle', 'rb') as f:
        pos_ids, neg_ids = pickle.load(f)
    f.close()

    with open('./data/hospitalization_data_1year.pickle', 'rb') as f:
        dt = pickle.load(f)
    f.close()

    dt_pos = dt[dt['ptid'].isin(pos_ids)]
    dt_neg = dt[dt['ptid'].isin(neg_ids)]
    dt_pos['y'] = 1
    dt_neg['y'] = 1
    train_ids, valid_ids, test_ids = split_train_validate_test(pos_ids, neg_ids)

