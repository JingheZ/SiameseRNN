"""
Prepare data for experiments
"""

import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import time
from sklearn import preprocessing


def split_train_validate_test(pos, neg):
    ids = np.array(list(pos) + list(neg))
    ys = [1] * len(pos) + [0] * len(neg)
    rs = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.2, random_state=1)
    train_ids = []
    test_ids = []
    valid_ids = []
    for train_ind, test_ind in rs.split(ids, ys):
        train_ids, test_ids = ids[train_ind], ids[test_ind]
    # valid_pos = set(pos).difference(set(train_ids).union(set(test_ids)))
    # rest_neg = set(neg).difference(set(train_ids).union(set(test_ids)))
    # valid_neg = random.sample(list(rest_neg), len(valid_pos))
    # valid_ids = list(valid_pos) + list(valid_neg)
        valid_ids = set(ids).difference(set(train_ids).union(set(test_ids)))
    return train_ids, valid_ids, test_ids


def find_previous_IP(dt):
    dt = dt[['ptid', 'cdrIPorOP']].drop_duplicates()
    dt_ipinfo = dt.groupby('ptid')['cdrIPorOP'].apply(list)
    return dt_ipinfo


def group_items_byadmmonth(dt, l):
    dt['adm'] = dt['adm_month'].apply(lambda x: int(x / l))
    dt = dt[['ptid', 'itemid', 'adm']].drop_duplicates().groupby(['ptid', 'adm'])['itemid'].apply(list)
    dt = dt.reset_index()
    pt_dict = {}
    for i in dt.index:
        ptid = dt['ptid'].loc[i]
        if not pt_dict.__contains__(ptid):
            pt_dict[ptid] = [[]] * int(12/l)
        adm = dt['adm'].loc[i]
        items = dt['itemid'].loc[i]
        pt_dict[ptid][adm] = items
    return pt_dict


def get_counts_subwindow(df, l):
    counts0 = df[['ptid', 'itemcat', 'adm']].drop_duplicates().groupby(['ptid', 'itemcat', 'adm']).size().unstack('itemcat')
    counts0.reset_index(inplace=True)
    dt = counts0[counts0['adm'] == 0]
    del dt['adm']
    dt.columns = ['ptid'] + ['t0_' + x for x in dt.columns[1:]]
    for j in range(1, int(12/l)):
        cts = counts0[counts0['adm'] == j]
        del cts['adm']
        cts.columns = ['ptid'] + ['t' + str(j) + '_' + x for x in cts.columns[1:]]
        dt = pd.merge(dt, cts, on='ptid', how='outer')
    dt.index = dt['ptid']
    del dt['ptid']
    dt.fillna(0, inplace=True)
    dt[dt > 0] = 1
    return dt


def pts_with_more_items(dt1, thres, l):
    dt1['adm'] = dt1['adm_month'].apply(lambda x: int(x/l))
    # del dt1['adm_month']
    dt = dt1[['ptid', 'itemid', 'adm']].drop_duplicates().groupby(['ptid', 'adm'])['itemid'].apply(list)
    dt = dt.reset_index()
    dt['length'] = dt['itemid'].apply(lambda x: len(x))
    # dt['length'].describe()
    # remove patients with items nums in 95%+ quantile
    pts_100 = dt[dt['length'] > thres]
    ids = set(pts_100['ptid'].values)
    return ids


if __name__ == '__main__':
    # =============== learn item embedding ================================
    with open('./data/visit_items_for_w2v.pickle', 'rb') as f:
        docs = pickle.load(f)
    f.close()

    # run the skip-gram w2v model
    size = 100
    window = 100
    min_count = 100
    workers = 28
    iter = 5
    sg = 1 # skip-gram:1; cbow: 0
    model_path = './results/w2v_size' + str(size) + '_window' + str(window) + '_sg' + str(sg)
    # if os.path.exists(model_path):
    #     model = Word2Vec.load(model_path)
    # else:
    # a = time.time()
    # model = Word2Vec(docs, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter)
    # model.save(model_path)
    # b = time.time()
    # print('training time (mins): %.3f' % ((b - a) / 60))

    # load model
    model = Word2Vec.load(model_path)
    vocab = list(model.wv.vocab.keys())
    # c = vocab[1]
    # sims = model.most_similar(c)
    # print(c)
    # print(sims)

    # =============== prepare training, validate, and test data ==============
    with open('./data/hospitalization_data_pos_neg_ids_v0.pickle', 'rb') as f:
        pos_ids, neg_ids = pickle.load(f)
    # f.close()

    with open('./data/hospitalization_data_1year.pickle', 'rb') as f:
        dt = pickle.load(f)
    # f.close()

    dt_pos = dt[dt['ptid'].isin(pos_ids)]
    dt_neg = dt[dt['ptid'].isin(neg_ids)]
    dt1 = pd.concat([dt_pos, dt_neg], axis=0)

    # dt1 = dt_pos
    item_cts = dt1[['ptid', 'itemid']].drop_duplicates().groupby('itemid').count()
    item_cts_100 = item_cts[item_cts['ptid'] > 50]
    dt1 = dt1[dt1['itemid'].isin(vocab)]
    dt1 = dt1[dt1['itemid'].isin(item_cts_100.index.tolist())]

    ids1 = pts_with_more_items(dt1, 105, 1)
    ids2 = pts_with_more_items(dt1, 116, 2)
    ids3 = pts_with_more_items(dt1, 125, 3)
    ids = ids1.union(ids2).union(ids3)
    dt1 = dt1[~dt1['ptid'].isin(ids)]
    ptids = set(dt1['ptid'].values)

    print('original_total_pts %i' % (len(pos_ids) + len(neg_ids))) # 103363
    print('updated_total_pts after excluding rare events %i' % len(ptids)) # 98202
    print('original_pos_pts %i' % len(pos_ids)) # 10416
    print('original_neg_pts %i' % len(neg_ids)) # 92947
    pos_ids = list(set(pos_ids).intersection(ptids))
    neg_ids = list(set(neg_ids).intersection(ptids))
    print('updated_pos_pts %i' % len(pos_ids)) # 8918
    print('updated_neg_pts %i' % len(neg_ids)) # 89284
    ptids = list(pos_ids) + list(neg_ids)
    train_ids, valid_ids, test_ids = split_train_validate_test(pos_ids, neg_ids)
    with open('./data/hospitalization_train_validate_test_ids.pickle', 'wb') as f:
        pickle.dump([train_ids, valid_ids, test_ids], f)
    f.close()

    # get IP info
    dt_ipinfo = find_previous_IP(dt1)
    train_ip = [1 if 'IP' in dt_ipinfo.loc[pid] else 0 for pid in train_ids]
    validate_ip = [1 if 'IP' in dt_ipinfo.loc[pid] else 0 for pid in valid_ids]
    test_ip = [1 if 'IP' in dt_ipinfo.loc[pid] else 0 for pid in test_ids]
    # get demographics info:
    with open('./data/orders_pt_info.pickle', 'rb') as f:
        pt_info_orders = pickle.load(f)
    f.close()
    #genders
    pt_info_orders = pt_info_orders[pt_info_orders['sex'].isin(['F', 'M'])]
    pt_info_orders['gender'] = pt_info_orders['sex'].map({'F': 1, 'M': 0})
    genders = pt_info_orders[['ptid', 'gender']].drop_duplicates().groupby('ptid').first()
    genders = genders['gender'].to_dict()
    train_genders = [genders[pid] for pid in train_ids]
    validate_genders = [genders[pid] for pid in valid_ids]
    test_genders = [genders[pid] for pid in test_ids]
    # ages
    ages = pt_info_orders[['ptid', 'age']].drop_duplicates().groupby('ptid').min()
    ages = ages['age'].to_dict()
    # ptids = list(pos_ids) + list(neg_ids)
    # ptids = train_ids
    ages_pts = [ages[pid] for pid in ptids]
    ages_scaled = preprocessing.scale(ages_pts)
    ages_scaled_df = pd.DataFrame([ptids, list(ages_scaled)])
    ages_scaled_df = ages_scaled_df.transpose()
    ages_scaled_df.columns = ['ptid', 'age']
    ages_scaled_df.index = ages_scaled_df['ptid']
    del ages_scaled_df['ptid']
    ages_scaled = ages_scaled_df['age'].to_dict()
    train_ages = [ages_scaled[pid] for pid in train_ids]
    validate_ages = [ages_scaled[pid] for pid in valid_ids]
    test_ages = [ages_scaled[pid] for pid in test_ids]

    # save static data: demographic and previous IP
    with open('./data/hospitalization_train_data_demoip.pickle', 'wb') as f:
        pickle.dump([train_genders, train_ages, train_ip], f)
    f.close()

    with open('./data/hospitalization_validate_data_demoip.pickle', 'wb') as f:
        pickle.dump([validate_genders, validate_ages, validate_ip], f)
    f.close()

    with open('./data/hospitalization_test_data_demoip.pickle', 'wb') as f:
        pickle.dump([test_genders, test_ages, test_ip], f)
    f.close()

    # get response
    train_y = [1 if pid in pos_ids else 0 for pid in train_ids]
    validate_y = [1 if pid in pos_ids else 0 for pid in valid_ids]
    test_y = [1 if pid in pos_ids else 0 for pid in test_ids]

    # get the counts of clinical events
    with open('./data/ccs_codes_all_item_categories.pickle', 'rb') as f:
        item_cats = pickle.load(f)
    f.close()
    itemdict = item_cats['cat'].to_dict()
    dt1['itemcat'] = [itemdict[x] for x in dt1['itemid'].values.tolist()]
    dt3 = dt1[['ptid', 'adm', 'itemcat']].drop_duplicates()
    # get counts over the entire observation window
    counts = dt3[['ptid', 'itemcat']].groupby(['ptid', 'itemcat']).size().unstack('itemcat').fillna(0)
    counts.reset_index(inplace=True)
    counts = counts[counts['ptid'].isin(ptids)]
    counts_ptids = counts['ptid'].values
    del counts['ptid']
    counts_scaled = preprocessing.scale(counts.values, axis=0)
    counts_scaled_dict = dict(zip(counts_ptids, counts_scaled.tolist()))
    train_cts = [counts_scaled_dict[pid] for pid in train_ids]
    validate_cts = [counts_scaled_dict[pid] for pid in valid_ids]
    test_cts = [counts_scaled_dict[pid] for pid in test_ids]
    # save counts of items
    with open('./data/hospitalization_train_data_cts.pickle', 'wb') as f:
        pickle.dump([train_cts, train_y], f)
    f.close()

    with open('./data/hospitalization_validate_data_cts.pickle', 'wb') as f:
        pickle.dump([validate_cts, validate_y], f)
    f.close()

    with open('./data/hospitalization_test_data_cts.pickle', 'wb') as f:
        pickle.dump([test_cts, test_y], f)
    f.close()

    with open('./data/hospitalization_cts_columns.pickle', 'wb') as f:
        pickle.dump(counts.columns.tolist(), f)
    f.close()

    # ============================ by time interval data =============================================
    for l in [1, 2, 3]:
        # get the itemids by month
        dt2 = group_items_byadmmonth(dt1, l)
        train = [dt2[pid] for pid in train_ids]
        validate = [dt2[pid] for pid in valid_ids]
        test = [dt2[pid] for pid in test_ids]

        # save data of items in each month
        with open('./data/hospitalization_train_data_by_' + str(l) + 'month.pickle', 'wb') as f:
            pickle.dump([train, train_y], f)
        f.close()

        with open('./data/hospitalization_validate_data_by_' + str(l) + 'month.pickle', 'wb') as f:
            pickle.dump([validate, validate_y], f)
        f.close()

        with open('./data/hospitalization_test_data_by_' + str(l) + 'month.pickle', 'wb') as f:
            pickle.dump([test, test_y], f)
        f.close()

        # get the counts of each month in the window
        counts_sub = get_counts_subwindow(dt3, l)
        counts_sub['ptid'] = counts_sub.index.values
        counts_sub = counts_sub[counts_sub['ptid'].isin(ptids)]
        counts_sub_ptids = counts_sub['ptid'].values
        del counts_sub['ptid']
        counts_sub_dict = dict(zip(counts_sub_ptids, counts_sub.values.tolist()))
        train_sub_cts = [counts_sub_dict[pid] for pid in train_ids]
        validate_sub_cts = [counts_sub_dict[pid] for pid in valid_ids]
        test_sub_cts = [counts_sub_dict[pid] for pid in test_ids]
        # save
        with open('./data/hospitalization_train_data_sub_cts_by_' + str(l) + 'month.pickle', 'wb') as f:
            pickle.dump([train_sub_cts, train_y], f)
        f.close()

        with open('./data/hospitalization_validate_data_sub_cts_by_' + str(l) + 'month.pickle', 'wb') as f:
            pickle.dump([validate_sub_cts, validate_y], f)
        f.close()

        with open('./data/hospitalization_test_data_sub_cts_by_' + str(l) + 'month.pickle', 'wb') as f:
            pickle.dump([test_sub_cts, test_y], f)
        f.close()

        with open('./data/hospitalization_cts_sub_columns_by_' + str(l) + 'month.pickle', 'wb') as f:
            pickle.dump(counts_sub.columns.tolist(), f)
        f.close()
