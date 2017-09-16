"""
Prepare data for experiments
"""

import pickle
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def split_train_validate_test(pos, neg):
    ids = list(pos) + list(neg)
    ys = [1] * len(pos) + [0] * len(neg)
    rs = StratifiedShuffleSplit(n_splits=1, train_size=0.65, test_size=0.2, random_state=1)
    train_ids = []
    test_ids = []
    valid_ids = []
    for train_ind, test_ind in rs.split(ids, ys):
        train_ids, test_ind = ids[train_ind], ids[test_ind]
        valid_ids = set(ids).difference(set(train_ids).union(set(test_ind)))
    return train_ids, test_ids, valid_ids


if __name__ == '__main__':

    with open('./data/hospitalization_data_pos_neg_ids.pickle', 'rb') as f:
        pos_ids, neg_ids = pickle.load(f)
    f.close()

    train_ids, valid_ids, test_ids = split_train_validate_test(pos_ids, neg_ids)