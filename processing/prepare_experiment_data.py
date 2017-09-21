"""
Prepare data for experiments
"""

import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
# import time


def split_train_validate_test(pos, neg):
    ids = np.array(list(pos) + list(neg))
    ys = [1] * len(pos) + [0] * len(neg)
    rs = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.2, random_state=1)
    train_ids = []
    test_ids = []
    valid_ids = []
    for train_ind, test_ind in rs.split(ids, ys):
        train_ids, test_ids = ids[train_ind], ids[test_ind]
        valid_ids = set(ids).difference(set(train_ids).union(set(test_ids)))
    return train_ids, valid_ids, test_ids


def find_previous_IP(dt):
    dt = dt[['ptid', 'cdrIPorOP']].drop_duplicates()
    dt_ipinfo = dt.groupby('ptid')['cdrIPorOP'].apply(list)
    return dt_ipinfo


def group_items_byadmmonth(dt):
    dt = dt[['ptid', 'itemid', 'adm_month']].groupby(['ptid', 'adm_month'])['itemid'].apply(list)
    dt = dt.reset_index()
    pt_dict = {}
    for i in dt.index:
        ptid = dt['ptid'].loc[i]
        if not pt_dict.__contains__(ptid):
            pt_dict[ptid] = [[]] * 12
        adm = dt['adm_month'].loc[i]
        items = dt['itemid'].loc[i]
        pt_dict[ptid][adm] = items
    return pt_dict


if __name__ == '__main__':
    # # =============== learn item embedding ================================
    # with open('./data/visit_items_for_w2v.pickle', 'rb') as f:
    #     docs = pickle.load(f)
    # f.close()
    #
    # # run the skip-gram w2v model
    size = 100
    window = 100
    # min_count = 100
    # workers = 28
    # iter = 5
    sg = 1 # skip-gram:1; cbow: 0
    model_path = './results/w2v_size' + str(size) + '_window' + str(window) + '_sg' + str(sg)
    # # if os.path.exists(model_path):
    # #     model = Word2Vec.load(model_path)
    # # else:
    # a = time.time()
    # model = Word2Vec(docs, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter)
    # model.save(model_path)
    # b = time.time()
    # print('training time (mins): %.3f' % ((b - a) / 60))
    #
    # load model
    model = Word2Vec.load(model_path)
    vocab = list(model.wv.vocab.keys())
    # c = vocab[1]
    # sims = model.most_similar(c)
    # print(c)
    # print(sims)

    # =============== prepare training, validate, and test data ==============
    with open('./data/hospitalization_data_pos_neg_ids.pickle', 'rb') as f:
        pos_ids, neg_ids = pickle.load(f)
    f.close()

    with open('./data/hospitalization_data_1year.pickle', 'rb') as f:
        dt = pickle.load(f)
    f.close()

    dt_pos = dt[dt['ptid'].isin(pos_ids)]
    dt_neg = dt[dt['ptid'].isin(neg_ids)]
    dt1 = pd.concat([dt_pos, dt_neg], axis=0)

    # dt1 = dt_pos
    item_cts = dt1[['ptid', 'itemid']].drop_duplicates().groupby('itemid').count()
    item_cts_100 = item_cts[item_cts['ptid'] > 50]
    dt1 = dt1[dt1['itemid'].isin(vocab)]
    dt1 = dt1[dt1['itemid'].isin(item_cts_100.index.tolist())]

    dt = dt1[['ptid', 'itemid', 'adm_month']].groupby(['ptid', 'adm_month'])['itemid'].apply(list)
    dt = dt.reset_index()
    dt['length'] = dt['itemid'].apply(lambda x: len(x))
    # remove patients with more than 115 items in a month
    pts_115 = dt[dt['length'] > 115]
    dt1 = dt1[~dt1['ptid'].isin(set(pts_115['ptid'].values))]

    dt2 = group_items_byadmmonth(dt1)
    dt_ipinfo = find_previous_IP(dt1)
    ptids = set(dt1['ptid'].values)

    print('original_total_pts %i' % (len(pos_ids) + len(neg_ids))) # 73877
    print('updated_total_pts after excluding rare events %i' % len(ptids)) # 69755
    print('original_pos_pts %i' % len(pos_ids)) # 3677
    print('original_neg_pts %i' % len(neg_ids)) # 70200
    pos_ids = list(set(pos_ids).intersection(ptids))
    neg_ids = list(set(neg_ids).intersection(ptids))
    print('updated_pos_pts %i' % len(pos_ids)) # 3025
    print('updated_neg_pts %i' % len(neg_ids)) # 66730

    train_ids, valid_ids, test_ids = split_train_validate_test(pos_ids, neg_ids)
    with open('./data/hospitalization_train_validate_test_ids.pickle', 'wb') as f:
        pickle.dump([train_ids, valid_ids, test_ids], f)
    f.close()
    train = [dt2[pid] for pid in train_ids]
    validate = [dt2[pid] for pid in valid_ids]
    test = [dt2[pid] for pid in test_ids]
    train_ip = [1 if 'IP' in dt_ipinfo.loc[pid] else 0 for pid in train_ids]
    validate_ip = [1 if 'IP' in dt_ipinfo.loc[pid] else 0 for pid in valid_ids]
    test_ip = [1 if 'IP' in dt_ipinfo.loc[pid] else 0 for pid in test_ids]
    train_y = [1 if pid in pos_ids else 0 for pid in train_ids]
    validate_y = [1 if pid in pos_ids else 0 for pid in valid_ids]
    test_y = [1 if pid in pos_ids else 0 for pid in test_ids]

    with open('./data/hospitalization_train_data.pickle', 'wb') as f:
        pickle.dump([train, train_ip, train_y], f)
    f.close()

    with open('./data/hospitalization_validate_data.pickle', 'wb') as f:
        pickle.dump([validate, validate_ip, validate_y], f)
    f.close()

    with open('./data/hospitalization_test_data.pickle', 'wb') as f:
        pickle.dump([test, test_ip, test_y], f)
    f.close()