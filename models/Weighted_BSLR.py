"""
Transfer learning for comorbid risk prediction
1. boosted or bagged SGL
2. significance: used domain adaptation for comorbid risk prediction; considers temporal info; works for small samples
"""

import pandas as pd
import random
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.utils import shuffle
from operator import itemgetter


def split_target_data(ptids, ratio):
    train_ids = []
    test_ids = []
    rs = ShuffleSplit(n_splits=1, test_size=ratio, random_state=1)
    for train_index, test_index in rs.split(ptids):
        train_ids, test_ids = ptids[train_index], ptids[test_index]
    return train_ids, test_ids


def get_counts_by_class(df, y, thres=50):
    def filter_rare_columns(data, thres):
        cols = data.columns
        num = len(data)
        cols_updated = []
        for i in cols:
            if 0 in data[i].value_counts(dropna=False).index:
                ct = data[i].value_counts(dropna=False)[0]
                if num - ct > thres:
                    cols_updated.append(i)
            else:
                cols_updated.append(i)
        data = data[cols_updated]
        return data
    df = df[['ptid', 'vid', 'dxcat']].drop_duplicates()
    counts = df[['ptid', 'dxcat']].groupby(['ptid', 'dxcat']).size().unstack('dxcat').fillna(0)
    if thres > 0:
        counts = filter_rare_columns(counts, thres)
    counts['response'] = y
    return counts


def create_subwindows(df, c=1):
    cols = df.columns
    if c > 0:
        if 'gap_ckd' not in cols:
            df = df[['ptid', 'dxcat', 'gap_dm']].drop_duplicates()
            vals = [min(int(-x / 24 / 60 / 30), 11) for x in df['gap_dm']]
            df['subw'] = [int(x / c) for x in vals]
        elif 'gap_dm' not in cols:
            df = df[['ptid', 'dxcat', 'gap_ckd']].drop_duplicates()
            vals = [max(1, 12 - int((x / 24 / 60) / 30)) for x in df['gap_ckd']]
            df['subw'] = [int((x - 1) / c) for x in vals]
        else:
            df = df[['ptid', 'dxcat', 'gap_ckd']].drop_duplicates()
            vals = [max(1, 12 - int((x / 24 / 60 - 180) / 30)) for x in df['gap_ckd']]
            df['subw'] = [int((x - 1) / c) for x in vals]
    else:
        df.sort(['ptid', 'adm_date', 'dxcat'], ascending=[1, 1, 1], inplace=True)
        df = df[['ptid', 'dxcat', 'adm_date']].drop_duplicates()
    return df
#
# def create_subwindows(df, c=1):
#     cols = df.columns
#     if c > 0:
#         if 'gap_dm' in cols:
#             df = df[['ptid', 'dxcat', 'gap_dm']].drop_duplicates()
#             vals = [min(int(-x / 24 / 60 / 30), 11) for x in df['gap_dm']]
#             df['subw'] = [int(x / c) for x in vals]
#         else:
#             df = df[['ptid', 'dxcat', 'adm_date']].drop_duplicates()
#             vals = [min(int(x / 24 / 60 / 30), 11) for x in df['adm_date']]
#             df['subw'] = [int(x / c) for x in vals]
#     else:
#         df.sort(['ptid', 'adm_date', 'dxcat'], ascending=[1, 1, 1], inplace=True)
#         df = df[['ptid', 'dxcat', 'adm_date']].drop_duplicates()
#     return df


def get_counts_subwindow(df, y, vars, c):
    def get_counts_one_window(counts0, j):
        cts = counts0[counts0['subw'] == j]
        del cts['subw']
        cts.columns = ['ptid'] + ['t' + str(j) + '_' + 'cat' + k for k in cts.columns[1:]]
        return cts

    df = df[df['dxcat'].isin(vars)]
    df = create_subwindows(df, c)
    counts0 = df[['ptid', 'dxcat', 'subw']].groupby(['ptid', 'dxcat', 'subw']).size().unstack('dxcat')
    counts0.reset_index(inplace=True)
    dt = get_counts_one_window(counts0, 0)
    for j in range(1, max(0, int(12/c))):
        cts = get_counts_one_window(counts0, j)
        dt = pd.merge(dt, cts, on='ptid', how='outer')
    dt['response'] = y
    dt.index = dt['ptid']
    del dt['ptid']
    dt.fillna(0, inplace=True)
    return dt


def split_shuffle_train_test_sets(train_ids, test_ids, valid_ids, X, y):
    train_x, test_x, valid_x = X.ix[train_ids], X.ix[test_ids], X.ix[valid_ids]
    train_y, test_y, valid_y = y.ix[train_ids], y.ix[test_ids], y.ix[valid_ids]
    train_x, train_y = shuffle(train_x, train_y, random_state=5)
    return train_x, train_y, test_x, test_y, valid_x, valid_y


def tune_proba_threshold_pred(pred_proba, y, b):
    results = []
    for t in np.arange(0, 1, 0.01):
        res = [1 if p > t else 0 for p in pred_proba]
        if b != 'auc':
            f1 = metrics.fbeta_score(y, res, beta=b)
            results.append((t, f1))
        else:
            auc0 = metrics.roc_auc_score(y, res)
            results.append((t, auc0))
    threshold = max(results, key=itemgetter(1))[0]
    return threshold


def make_prediction_and_tuning(train_x, train_y, test_x, test_y, param):
    if param[3] == 'rf':
        clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=param[1], random_state=0,
                                     min_samples_leaf=param[4])
        # clf = LogisticRegression(penalty='l1', C=param[0], n_jobs=param[1], random_state=0)
        clf.fit(train_x, train_y)
        pred_test = clf.predict_proba(test_x)
        test_pred_proba = [i[1] for i in pred_test]
        # threshold tuning with f measure
        threshold = tune_proba_threshold_pred(test_pred_proba, test_y, param[2])
        print('Threshold %.3f tuned with f measure' % (threshold))

    else:
        clf = LogisticRegression(penalty='l1', C=param[0], n_jobs=param[1], random_state=0)
        clf.fit(train_x, train_y)
        pred_test = clf.predict_proba(test_x)
        test_pred_proba = [i[1] for i in pred_test]
        # threshold tuning with f measure
        threshold = tune_proba_threshold_pred(test_pred_proba, test_y, param[2])
        print('Threshold %.3f tuned with f measure' % (threshold))
    return clf, threshold


def get_transition_counts(dm, control, extra, vars):
    dm = create_subwindows(dm, 0)
    control = create_subwindows(control, 0)
    extra = create_subwindows(extra, 0)
    df = pd.concat([dm, control, extra], axis=0)
    df['dxcat'] = df['dxcat'].astype(int)
    df.sort(['ptid', 'adm_date', 'dxcat'], ascending=[1, 1, 1], inplace=True)
    df2 = df.groupby(['ptid', 'adm_date'])['dxcat'].apply(list)
    df3 = df2.reset_index()
    df4 = df3.groupby(['ptid'])['dxcat'].apply(list)
    seq_dict = df4.to_dict()
    vars_pairs = []
    for var in vars:
        pairs = [str(var) + 'to' + str(var2) for var2 in vars]
        vars_pairs += pairs
    transition_dict = {}
    for key, val in seq_dict.items():
        trans = dict.fromkeys(vars_pairs, 0)
        if len(val) > 1:
            for i in range(len(val[:-1])):
                for v2 in val[i + 1:]:
                    for vv0 in val[i]:
                        for vv2 in v2:
                            trans[str(vv0) + 'to' + str(vv2)] += 1
        transition_dict[key] = list(trans.values())
    transitions = pd.DataFrame.from_dict(transition_dict, orient='index')
    transitions.columns = vars_pairs
    return transitions


def create_sequence(df, s):
    df = create_subwindows(df, 0)
    df['dxcat'] = df['dxcat'].astype(int)
    # df.sort(['ptid', 'adm_date', 'dxcat'], ascending=[1, 1, 1], inplace=True)
    df2 = df.groupby(['ptid', 'adm_date'])['dxcat'].apply(tuple)
    df3 = df2.reset_index()
    df4 = df3.groupby(['ptid'])['dxcat'].apply(list)
    df4.to_csv('./data/seq_' + s + '.txt', header=None, index=False, sep=' ', mode='a')
    # # to replace redundant strings in linux:
    # sed -i 's/,)/)/g' ./data/seq_dm_train.txt
    # sed -i 's/"//g' ./data/seq_dm_train.txt
    # sed -i 's/\[//g' ./data/seq_dm_train.txt
    # sed -i 's/\]//g' ./data/seq_dm_train.txt
    # sed -i 's/),/ -1/g' ./data/seq_dm_train.txt
    # sed -i 's/(//g' ./data/seq_dm_train.txt
    # sed -i 's/,//g' ./data/seq_dm_train.txt
    # sed -i 's/)/ -2/g' ./data/seq_dm_train.txt

    # sed -i 's/,)/)/g' ./data/seq_control_train.txt
    # sed -i 's/"//g' ./data/seq_control_train.txt
    # sed -i 's/\[//g' ./data/seq_control_train.txt
    # sed -i 's/\]//g' ./data/seq_control_train.txt
    # sed -i 's/),/ -1/g' ./data/seq_control_train.txt
    # sed -i 's/(//g' ./data/seq_control_train.txt
    # sed -i 's/,//g' ./data/seq_control_train.txt
    # sed -i 's/)/ -2/g' ./data/seq_control_train.txt
    return df3


def get_seq_item_counts(seq_dm, seq_control, seq_extra, cooccur_list, mitmvisit_list):
    # get items occurred at the same time
    def get_count_one_itemset(seq, cs):
        c1 = cs[0]
        c2 = cs[1]
        if len(cs) == 2:
            ct1 = [1 if c1 in it and c2 in it else 0 for it in seq['dxcat'].values.tolist()]
            seq['cat' + str(c1) + '_' + str(c2)] = ct1
            count1 = seq[['ptid', 'cat' + str(c1) + '_' + str(c2)]].groupby('ptid').sum()
        else:
            c3 = cs[2]
            ct1 = [1 if c1 in it and c2 in it and c3 in it else 0 for it in seq['dxcat'].values.tolist()]
            seq['cat' + str(c1) + '_' + str(c2) + '_' + str(c3)] = ct1
            count1 = seq[['ptid', 'cat' + str(c1) + '_' + str(c2) + '_' + str(c3)]].groupby('ptid').sum()
        return count1
    seq = pd.concat([seq_dm[['ptid', 'dxcat']], seq_control[['ptid', 'dxcat']], seq_extra[['ptid', 'dxcat']]], axis=0)
    count_ab = get_count_one_itemset(seq, cooccur_list[0])
    for a in cooccur_list[1:]:
        countb = get_count_one_itemset(seq, a)
        count_ab = pd.concat([count_ab, countb], axis=1)

    # # get same item occurred in different visits
    # def get_count_two_visits(seq, c):
    #     ct1 = [1 if c in it else 0 for it in seq['dxcat'].values.tolist()]
    #     seq['var'] = ct1
    #     seq = seq[seq['var'] > 0]
    #     count1 = seq[['ptid', 'var']].groupby('ptid').count()
    #     count1['cat' + str(c) + 'to' + str(c)] = [i * (i - 1) * 0.5 for i in count1['var'].values]
    #     del count1['var']
    #     return count1
    # count_cd = get_count_two_visits(seq, mvisit_list[0])
    # for c in mvisit_list[1:]:
    #     countc = get_count_two_visits(seq, c)
    #     count_cd = pd.concat([count_cd, countc], axis=1)
    # count_abcd = pd.concat([count_ab, count_cd], axis=1)

    # get multiple item occurred in different visits
    def get_count_multiple_items_mvisits(seq2, c):
        def get_visit_ind_with_item(seq0, l):
            inds = []
            for v, vt in enumerate(seq0):
                fg = 1
                for c0 in l:
                    if c0 not in vt:
                        fg = 0
                        break
                if fg == 1:
                    inds.append(v)
            return inds
        def calculate_vcts(inds1, inds2):
            ct = 0
            for i in inds1:
                for j in inds2:
                    if i < j:
                        ct += 1
            return ct

        val = []
        for p in seq2.index:
            seq0 = seq2['dxcat'].loc[p]
            ct1 = get_visit_ind_with_item(seq0, c[0])
            ct2 = get_visit_ind_with_item(seq0, c[1])
            va = calculate_vcts(ct1, ct2)
            val.append(va)
        val = np.array(val)
        nm1 = '+'.join([str(c0) for c0 in c[0]])
        nm2 = '+'.join([str(c0) for c0 in c[1]])
        return val, [nm1 + 'to' + nm2]
    seq2 = seq.groupby(['ptid'])['dxcat'].apply(list)
    seq2 = seq2.reset_index()
    seq2.index = seq2['ptid']
    count_cd = seq2['ptid'].values
    nms = ['ptid']
    for c in mitmvisit_list:
        countc, nm0 = get_count_multiple_items_mvisits(seq2, c)
        count_cd = np.vstack((count_cd, countc))
        nms += nm0
    counts_cd = pd.DataFrame(count_cd)
    counts_cd = counts_cd.transpose()
    counts_cd.columns = nms
    counts_cd.index = counts_cd['ptid']
    del counts_cd['ptid']
    count_abcd = pd.concat([count_ab, counts_cd], axis=1)
    return count_abcd


def make_predictions(train_x, train_y, test_x, param):
    # train_x = train_x.as_matrix().astype(np.float)
    # test_x = test_x.as_matrix().astype(np.float)
    # train_y = train_y.as_matrix().astype(np.float)
    if param[2] == 'rf':
        clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=param[1], random_state=0,
                                     min_samples_leaf=50, min_samples_split=2)
        # clf = LogisticRegression(penalty='l1', C=param[0], n_jobs=param[1], random_state=0)
        clf.fit(train_x, train_y)
        pred_test = clf.predict_proba(test_x)
        test_pred_proba = [i[1] for i in pred_test]
    else:
        clf = LogisticRegression(penalty='l1', C=param[0], n_jobs=param[1], random_state=0)
        clf.fit(train_x, train_y)
        pred_test = clf.predict_proba(test_x)
        test_pred_proba = [i[1] for i in pred_test]
    return test_pred_proba


if __name__ == '__main__':
    # ===================== load data =====================================
    with open('./data/data_dm_ptids.pickle', 'rb') as f:
        data_dm, ptids_dm = pickle.load(f)
    f.close()
    with open('./data/data_ckd_ptids.pickle', 'rb') as f:
        data_ckd, ptids_ckd = pickle.load(f)
    f.close()
    #
    # with open('./data/data_chf_ptids.pickle', 'rb') as f:
    #     data_chf, ptids_chf = pickle.load(f)
    # f.close()

    data_dm_ckd = pd.merge(data_dm, data_ckd[['ptid', 'first_ckd_date']].drop_duplicates(), how='inner', left_on='ptid',
                           right_on='ptid')
    data_dm_ckd.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    data_dm_ckd['gap_dm_ckd'] = data_dm_ckd['first_ckd_date'] - data_dm_ckd['first_dm_date']
    ptids_dm_ckd = set(data_dm_ckd['ptid'])  # 4803 pts
    d0 = data_dm_ckd[data_dm_ckd['first_ckd_date'] >= 180 * 24 * 60]
    ptids_dm_ckd0 = set(d0['ptid'])  # 1329 pts
    d1 = d0[d0['gap_dm_ckd'] >= (180 + 360) * 24 * 60].drop_duplicates()
    ptids_dm_ckd1 = set(d1['ptid'])  # 561 pts

    # get the data for dm-ckd (target group):
    data_dm_ckd2 = data_dm_ckd[data_dm_ckd['ptid'].isin(ptids_dm_ckd1)]
    data_dm_ckd2['gap_ckd'] = data_dm_ckd2['first_ckd_date'] - data_dm_ckd2['adm_date']
    data_dm_ckd3 = data_dm_ckd2[data_dm_ckd2['gap_ckd'].between(180 * 24 * 60, 540 * 24 * 60)]
    ptids_dm_ckd3 = set(data_dm_ckd3['ptid'])  # 410 pts

    # get the dm data for training: 2.5 years of history after first dm: 1yr observation,
    # half year hold off, and 1yr prediction
    data_dm2 = data_dm[~data_dm['ptid'].isin(ptids_ckd)]
    data_dm2 = data_dm2[~data_dm2['ptid'].isin(ptids_dm_ckd)]
    data_dm3 = data_dm2[data_dm2['gap_dm'] <= -360 * 2.5 * 24 * 60]
    ptids_dm3 = set(data_dm3['ptid'])  # 6259 pts
    data_dm4 = data_dm[data_dm['ptid'].isin(ptids_dm3)]
    data_dm5 = data_dm4[data_dm4['gap_dm'].between(-360 * 24 * 60, 0)]
    ptids_dm5 = set(data_dm5['ptid'])  # 6259 pts

    # get the data as training of the ckd class -1.5 to -0.5 years prior to first ckd
    data_ckd2 = data_ckd[~data_ckd['ptid'].isin(ptids_dm)]
    data_ckd3 = data_ckd2[data_ckd2['gap_ckd'] > 360 * 1 * 24 * 60]
    ptids_ckd3 = set(data_ckd3['ptid'])  # 1022 pts
    data_ckd4 = data_ckd[data_ckd['ptid'].isin(ptids_ckd3)]
    data_ckd3 = data_ckd4[data_ckd4['adm_date'].between(0, 360 * 24 * 60)]
    ptids_ckd3 = set(data_ckd3['ptid'])  # 1022 pts

    # get preliminary features
    counts_dm = get_counts_by_class(data_dm5, 0, len(ptids_dm5) * 0.1)
    counts_ckd = get_counts_by_class(data_ckd3, 1, len(ptids_ckd3) * 0.1)
    counts_dmckd = get_counts_by_class(data_dm_ckd3, 1, len(ptids_dm_ckd3) * 0.1)
    counts = counts_dm.append(counts_ckd).append(counts_dmckd).fillna(0)
    prelim_features = set(counts.columns[:-1]) # 73
    # update datasets to exclude unselected features
    data_dm = data_dm5[data_dm5['dxcat'].isin(prelim_features)]
    data_ckd = data_ckd3[data_ckd3['dxcat'].isin(prelim_features)]
    data_dmckd = data_dm_ckd3[data_dm_ckd3['dxcat'].isin(prelim_features)]
    ptids_dm = list(set(data_dm['ptid'].values.tolist())) # 6259
    ptids_ckd = list(set(data_ckd['ptid'].values.tolist()))  # 981
    ptids_dmckd = list(set(data_dmckd['ptid'].values.tolist()))  # 401
    # get aggregated counts
    counts_dm = get_counts_by_class(data_dm, 0, 0)
    counts_ckd = get_counts_by_class(data_ckd, 1, 0)
    counts_dmckd = get_counts_by_class(data_dmckd, 1, 0)
    counts = counts_dm.append(counts_ckd).append(counts_dmckd).fillna(0)
    counts.columns = ['cat' + i for i in counts.columns[:-1]] + ['response']
    counts.to_csv('./data/comorbid_task_counts.csv')

    # get subw counts
    counts_sub_dm = get_counts_subwindow(data_dm, 0, prelim_features, 6)
    counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 6)
    counts_sub_dmckd = get_counts_subwindow(data_dmckd, 1, prelim_features, 6)
    counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    counts_sub.to_csv('./data/comorbid_task_counts_sub_by6momth.csv')

    counts_sub_dm = get_counts_subwindow(data_dm, 0, prelim_features, 4)
    counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 4)
    counts_sub_dmckd = get_counts_subwindow(data_dmckd, 1, prelim_features, 4)
    counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    counts_sub.to_csv('./data/comorbid_task_counts_sub_by4momth.csv')

    # counts_sub_dm = get_counts_subwindow(data_dm, 0, prelim_features, 2)
    # counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 2)
    # counts_sub_dmckd = get_counts_subwindow(data_dmckd, 1, prelim_features, 2)
    # counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    # counts_sub.to_csv('./data/comorbid_task_counts_sub_by2momth.csv')

    # =============== primary representation =============================================================
    counts = pd.read_csv('./data/comorbid_task_counts.csv')
    counts.index = counts['ptid'].astype(str)
    del counts['ptid']
    # baseline 1: aggregated count vector
    counts_x = counts[counts.columns[:-1]]
    counts_y = counts['response']
    features1 = counts_x.columns.tolist()

    # baseline 2: subw count vector
    counts_sub_x = counts_sub[counts_sub.columns[1:]]
    counts_sub_y = counts_sub['response']
    features2 = counts_sub_x.columns.tolist()

    # # baseline 3: mining sequence patterns
    # get the sequence by sub-windows
    seq_dm = create_sequence(data_dmckd, 'comorbid_risk_dmckd_train')
    seq_control = create_sequence(data_dm, 'comorbid_risk_control_train')
    seq_extra = create_sequence(data_dm, 'comorbid_risk_ckd_train')
    cooccur_list = [[49, 98], [53, 98], [49, 53], [49, 53, 98], [259, 663], [49, 663], [257, 259],
                    [133, 259], [95, 98], [58, 98], [49, 95], [49, 58], [49, 58, 98], [49, 95, 98]]
    # mvisit_list = [98, 259, 49, 53]
    mitmvisit_list = [[[98], [98]], [[259], [259]], [[49], [49]], [[53], [53]],
                      [[49], [98]], [[98], [49]], [[98], [49, 98]], [[49], [49, 98]],
                      [[49, 98], [49]], [[49, 98], [98]], [[49, 98], [49, 98]], [[98], [259]],
                      [[49], [259]], [[259], [49]], [[98], [257]], [[49], [257]], [[49], [211]],
                      [[49], [133]], [[53], [98]], [[49], [53]], [[53], [49]], [[53], [49, 53]],
                      [[53], [49, 98]], [[53, 98], [49]], [[53], [53, 98]], [[53, 98], [98]],
                      [[49], [49, 53]], [[49, 53], [49]], [[49], [259, 49]], [[49], [49, 53, 98]],
                      [[49, 53], [53]], [[49], [53, 98]], [[49, 53], [98]], [[49, 53, 98], [98]],
                      [[49, 53, 98], [49]], [[49, 53], [49, 53]], [[49, 98], [257]], [[49, 98], [259]]]

    counts_bpsb = get_seq_item_counts(seq_dm, seq_control, seq_extra, cooccur_list, mitmvisit_list)
    counts_bps = pd.concat([counts_bpsb, counts[['response']]], axis=1).fillna(0)
    counts_bps.to_csv('./data/comorbid_risk_counts_bps.csv')
    counts_bps_y = counts_bps['response']
    counts_bps_x = counts_bps
    del counts_bps_x['response']
    features3 = counts_bps_x.columns.tolist()

    # baseline 4: transitions
    counts_trans = get_transition_counts(data_dmckd, data_dm, data_ckd, prelim_features)
    counts_trans = pd.concat([counts_trans, counts], axis=1).fillna(0)
    counts_trans.to_csv('./data/comorbid_risk_counts_trans.csv')

    counts_trans_x = counts_trans[counts_trans.columns[:-1]]
    counts_trans_y = counts_trans['response']
    features4 = counts_trans.columns.tolist()[:-1]

    # ================ split train and testing data ========================================

    ratio = 1.5
    # randomly select 60% for training and 40% for testing from target group
    train_ids_dm_ckd, test_ids_dm_ckd = split_target_data(np.array(ptids_dmckd), 0.4)
    train_ids_dm_ckd, valid_ids_dm_ckd = split_target_data(np.array(train_ids_dm_ckd), 0.33)
    random.seed(5)
    test_ids_dm = random.sample(ptids_dm, int(len(test_ids_dm_ckd) * ratio))
    rest_dm_ptids = list(set(ptids_dm).difference(set(test_ids_dm)))
    test_ids = list(test_ids_dm) + list(test_ids_dm_ckd)
    pd.Series(test_ids).to_csv('./data/comorbid_risk_test_ids.csv', index=False)
    random.seed(5)
    valid_ids_dm = random.sample(rest_dm_ptids, int(len(valid_ids_dm_ckd) * ratio))
    rest_dm_ptids = list(set(rest_dm_ptids).difference(set(valid_ids_dm)))
    valid_ids = list(valid_ids_dm) + list(valid_ids_dm_ckd)
    pd.Series(valid_ids).to_csv('./data/comorbid_risk_validation_ids.csv', index=False)

    for r in [0, 1, 3, 6]:
        print(r)
        train_ids = list(train_ids_dm_ckd)
        num_ckd = int(r * len(train_ids_dm_ckd))
        if num_ckd < len(ptids_ckd):
            random.seed(5)
            train_ids_ckd = random.sample(ptids_ckd, num_ckd)
        else:
            train_ids_ckd = ptids_ckd
        random.seed(5)
        train_ids_dm = random.sample(rest_dm_ptids, int((len(train_ids_ckd) + len(train_ids_dm_ckd)) * ratio))
        # train_ids_dm = random.sample(rest_dm_ptids, num_ckd * ratio)
        train_ids += list(train_ids_dm) + list(train_ids_ckd)
        pd.Series(train_ids).to_csv('./data/comorbid_risk_train_ids_r' + str(r) + '.csv', index=False)

        r = 0
        test_ids = pd.read_csv('./data/comorbid_risk_test_ids.csv', header=None, dtype=object)
        test_ids = test_ids.values.flatten()
        train_ids = pd.read_csv('./data/comorbid_risk_train_ids_r0.csv', header=None, dtype=object)
        train_ids = train_ids.values.flatten()
        valid_ids = pd.read_csv('./data/comorbid_risk_validation_ids.csv', header=None, dtype=object)
        valid_ids = valid_ids.values.flatten()
        # baseline 1: freq
        counts = pd.read_csv('./data/comorbid_task_counts.csv')
        counts.index = counts['ptid'].astype(str)
        del counts['ptid']
        counts_x = counts[counts.columns[:-1]]
        counts_y = counts['response']
        features0 = counts.columns.tolist()[:-1]
        # baseline 2: BPS
        # seq_dm = create_sequence(data_dmckd[data_dmckd['ptid'].isin(train_ids)], 'comorbid_risk_dmckd_train')
        # seq_control = create_sequence(data_dm[data_dm['ptid'].isin(train_ids)], 'comorbid_risk_control_train')
        # seq_extra = create_sequence(data_dm, 'comorbid_risk_ckd_train')
        #
        # cooccur_list = [[49, 663], [257, 259], [133, 259], [133, 257], [133, 257, 259], [53, 98], [49, 98],
        #                 [49, 95], [49, 53], [49, 53, 98], [259, 663], [257, 663], [95, 98], [50, 98], [53, 95],
        #                 [50, 53], [53, 95, 98], [49, 53, 95], [49, 95, 98]]
        # # mvisit_list = [98, 259, 49, 53]
        # mitmvisit_list = [[[259], [259]], [[259], [257]], [[49], [259]], [[259], [49]], [[257], [257]],
        #                   [[49], [257]], [[49], [133]], [[98], [98]], [[53], [98]], [[98], [53]],
        #                   [[49], [98]], [[98], [49]], [[98], [49, 98]], [[53, 98], [98]], [[49], [49]],
        #                   [[49], [49, 98]], [[49, 98], [49]], [[49], [133, 49]], [[49], [257, 49]],
        #                   [[49, 98], [98]], [[49, 98], [49, 98]], [[49], [133, 259]], [[98], [259]], [[259], [98]],
        #                   [[259], [49, 98]], [[49], [256]], [[49], [211]], [[49], [205]], [[98], [53, 98]],
        #                   [[49], [95]], [[53], [53]], [[49], [53]], [[53], [49]], [[53], [53, 98]],
        #                   [[49], [49, 53]], [[49, 53], [49]], [[49], [259, 49]], [[49, 53], [53]],
        #                   [[49, 98], [98]], [[49, 98], [259]], [[49, 98], [49, 98]]]
        #
        #
        # # cooccur_list = [[49, 98], [53, 98], [49, 53], [49, 53, 98], [259, 663], [49, 663], [257, 259],
        # #                 [133, 259], [95, 98], [58, 98], [49, 95], [49, 58], [49, 58, 98], [49, 95, 98]]
        # # # # mvisit_list = [98, 259, 49, 53]
        # # mitmvisit_list = [[[98], [98]], [[259], [259]], [[49], [49]], [[53], [53]],
        # #                   [[49], [98]], [[98], [49]], [[98], [49, 98]], [[49], [49, 98]],
        # #                   [[49, 98], [49]], [[49, 98], [98]], [[49, 98], [49, 98]], [[98], [259]],
        # #                   [[49], [259]], [[259], [49]], [[98], [257]], [[49], [257]], [[49], [211]],
        # #                   [[49], [133]], [[53], [98]], [[49], [53]], [[53], [49]], [[53], [49, 53]],
        # #                   [[53], [49, 98]], [[53, 98], [49]], [[53], [53, 98]], [[53, 98], [98]],
        # #                   [[49], [49, 53]], [[49, 53], [49]], [[49], [259, 49]], [[49], [49, 53, 98]],
        # #                   [[49, 53], [53]], [[49], [53, 98]], [[49, 53], [98]], [[49, 53, 98], [98]],
        # #                   [[49, 53, 98], [49]], [[49, 53], [49, 53]], [[49, 98], [257]], [[49, 98], [259]]]
        #
        # counts_bpsb = get_seq_item_counts(seq_dm, seq_control, seq_extra, cooccur_list, mitmvisit_list)
        # counts_bps = pd.concat([counts_bpsb, counts], axis=1).fillna(0)
        # counts_bps.to_csv('./data/comorbid_risk_counts_bps.csv')
        counts_bps = pd.read_csv('./data/comorbid_risk_counts_bps.csv')
        counts_bps.columns = ['ptid'] + list(counts_bps.columns[1:])
        counts_bps.index = counts_bps['ptid'].astype(str)
        del counts_bps['ptid']
        counts_bps_x = counts_bps[counts_bps.columns[:-1]]
        counts_bps_y = counts_bps['response']
        features3 = counts_bps_x.columns.tolist()[:-1]

        # baseline 3: trans
        counts_trans = pd.read_csv('./data/comorbid_risk_counts_trans.csv')
        counts_trans.columns = ['ptid'] + list(counts_trans.columns[1:])
        counts_trans.index = counts_trans['ptid'].astype(str)
        del counts_trans['ptid']
        counts_trans_x = counts_trans[counts_trans.columns[:-1]]
        counts_trans_y = counts_trans['response']
        features4 = counts_trans.columns.tolist()[:-1]

        # baseline 4: single LSR
        counts_sub = pd.read_csv('./data/comorbid_task_counts_sub_by4momth.csv')
        counts_sub.index = counts_sub['ptid'].astype(str)
        del counts_sub['ptid']
        features5a = counts_sub.columns.tolist()[1:]
        features5_all = pd.read_csv('./data/sgl_coefs_alpha7_r0_bootstrap15.csv')
        i = features5_all.columns[1]
        features5_inds = features5_all[i]
        features5 = [features5a[j] for j in features5_inds.index if features5_inds.loc[j] != 0]
        counts_sgl_x = counts_sub[features5]
        counts_sgl_y = counts_sub['response']

        train_x0, train_y0, test_x0, test_y0 , valid_x0, valid_y0 = split_shuffle_train_test_sets(train_ids, test_ids,
                                                                                                  valid_ids, counts_x,
                                                                                                  counts_y)
        # # #
        # train_x1, train_y1, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids, test_ids, counts_sub_x, counts_sub_y)
        #
        train_x2, train_y2, test_x2, test_y2, valid_x2, valid_y2 = split_shuffle_train_test_sets(train_ids, test_ids,
                                                                                                 valid_ids, counts_bps_x,
                                                                                                 counts_bps_y)
        train_x3, train_y3, test_x3, test_y3, valid_x3, valid_y3 = split_shuffle_train_test_sets(train_ids, test_ids,
                                                                                                 valid_ids, counts_trans_x,
                                                                                                 counts_trans_y)

        train_x4, train_y4, test_x4, test_y4, valid_x4, valid_y4 = split_shuffle_train_test_sets(train_ids, test_ids,
                                                                                                 valid_ids, counts_sgl_x,
                                                                                                 counts_sgl_y)

        clf0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, valid_x0, valid_y0, [20, 15, 1, 'rf', 1])
        clf0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, valid_x0, valid_y0, [0.05, 15, 1, 'lr'])
        # threshold: lr: 0.44; rf size = 20 (0.49); rf size = 50 (0.40);

        clf2, results_by_f2 = make_prediction_and_tuning(train_x2, train_y2, valid_x2, valid_y2, [20, 15, 1, 'rf', 1])
        clf2, results_by_f2 = make_prediction_and_tuning(train_x2, train_y2, valid_x2, valid_y2, [0.05, 15, 1, 'lr'])
        # threshold: lr: 0.53; rf size = 20 (0.55); rf size = 50 (0.66);

        clf3, results_by_f3 = make_prediction_and_tuning(train_x3, train_y3, valid_x3, valid_y3, [20, 15, 1, 'rf', 1])
        clf3, results_by_f3 = make_prediction_and_tuning(train_x3, train_y3, valid_x3, valid_y3, [0.05, 15, 1, 'lr'])
        # threshold: lr: 0.43; rf size = 20 (0.45); rf size = 50 (0.35);

        clf4, results_by_f4 = make_prediction_and_tuning(train_x4, train_y4, valid_x4, valid_y4, [20, 15, 1, 'rf', 50])
        clf4, results_by_f4 = make_prediction_and_tuning(train_x4, train_y4, valid_x4, valid_y4, [0.01, 15, 1, 'lr'])
        # threshold: lr: 0.48; rf size = 20 (0.36); rf size = 50 (0.37);
        for n in [10, 20, 30, 50, 100, 150, 500]:
            test_proba0a = make_predictions(train_x0, train_y0, test_x0, [n, 15, 'rf'])
            test_proba0b = make_predictions(train_x0, train_y0, test_x0, [200, 15, 'lr'])
            test_proba0c = make_predictions(train_x0, train_y0, test_x0, [0.05, 15, 'lr'])
            # #
            # test_proba1a = make_predictions(train_x1, train_y1, test_x1, [10, 15, 'rf'])
            # test_proba1b = make_predictions(train_x1, train_y1, test_x1, [200, 15, 'lr'])
            # test_proba1c = make_predictions(train_x1, train_y1, test_x1, [0.05, 15, 'lr'])
            # # # #
            test_proba2a = make_predictions(train_x2, train_y2, test_x2, [n, 15, 'rf'])
            test_proba2b = make_predictions(train_x2, train_y2, test_x2, [200, 15, 'lr'])
            test_proba2c = make_predictions(train_x2, train_y2, test_x2, [0.05, 15, 'lr'])
            # # # # #
            test_proba3a = make_predictions(train_x3, train_y3, test_x3, [n, 15, 'rf'])
            test_proba3b = make_predictions(train_x3, train_y3, test_x3, [200, 15, 'lr'])
            test_proba3c = make_predictions(train_x3, train_y3, test_x3, [0.05, 15, 'lr'])
            # # # #
            test_proba4a = make_predictions(train_x4, train_y4, test_x4, [n, 15, 'rf'])
            test_proba4b = make_predictions(train_x4, train_y4, test_x4, [200, 15, 'lr'])
            test_proba4c = make_predictions(train_x4, train_y4, test_x4, [0.05, 15, 'lr'])
            # # # #
            test_proba = pd.DataFrame([test_proba0a, test_proba0b, test_proba0c, test_y0.values.tolist(),
                                       # test_proba1a, test_proba1b, test_proba1c, test_y1.values.tolist(),
                                       test_proba2a, test_proba2b, test_proba2c, test_y2.values.tolist(),
                                       test_proba3a, test_proba3b, test_proba3c, test_y3.values.tolist(),
                                       test_proba4a, test_proba4b, test_proba4c, test_y4.values.tolist()])
            test_proba = test_proba.transpose()
            test_proba.columns = ['b1_rf', 'b1_lr', 'b1_lasso', 'b1_response',
                                  'b3_rf', 'b3_lr', 'b3_lasso', 'b3_response', 'b4_rf', 'b4_lr', 'b4_lasso', 'b4_response',
                                  'b5_rf', 'b5_lr', 'b5_lasso', 'b5_response']
            test_proba.to_csv('./data/comorbid_risk_test_proba_baselines_r' + str(r) + '_bs' + str(n) + '.csv', index=False)

        # # # data_dm4[data_dm4['ptid'] == '769052'].to_csv('./data/example_dmpt.csv') # rf predicted proba: 0.782
        # # # data_control4[data_control4['ptid'] =='1819093'].to_csv('./data/example_controlpt.csv') # rf predicted proba: 0.033

        for p in range(100):
            random.seed(p)
            sp = random.choices(list(enumerate(train_ids)), k=len(train_ids))
            sp_ptids = [i[1] for i in sp]
            sp_inds = [i[0] for i in sp]
            sp_dt = pd.DataFrame([sp_ptids, sp_inds]).transpose()
            sp_dt.columns = ['ptid', 'ind']
            sp_dt.to_csv('./data/comorbid_risk_train_ids_bootstrap' + str(p) + '.csv', index=False)



        features5_all = pd.read_csv('./data/sgl_coefs_alpha7_r0_bootstrap28.csv')
        i = features5_all.columns[1]
        features5_inds = features5_all[i]
        features5 = [features5a[j] for j in features5_inds.index if features5_inds.loc[j] != 0]
        counts_sgl_x = counts_sub[features5]
        counts_sgl_y = counts_sub['response']
        train_x4, train_y4, test_x4, test_y4, valid_x4, valid_y5 = split_shuffle_train_test_sets(train_ids, test_ids, valid_ids, counts_sgl_x, counts_sgl_y)

        for n in [10, 20, 30, 50, 100, 150, 500, 1000]:
            # # # #
            test_proba4a = make_predictions(train_x4, train_y4, test_x4, [n, 15, 'rf'])
            test_proba4b = make_predictions(train_x4, train_y4, test_x4, [200, 15, 'lr'])
            test_proba4c = make_predictions(train_x4, train_y4, test_x4, [0.01, 15, 'lr'])
            # # # #
            test_proba = pd.DataFrame([test_proba4a, test_proba4b, test_proba4c, test_y4.values.tolist()])
            test_proba = test_proba.transpose()
            test_proba.columns = ['b5_rf', 'b5_lr', 'b5_lasso', 'b5_response']
            test_proba.to_csv('./data/comorbid_risk_test_proba_baseline_LSR_r' + str(r) + '_bs' + str(n) + '.csv', index=False)

        # get the features of the model with the highest weight in the proposed framework
        features5_all = pd.read_csv('./data/sgl_coefs_alpha7_r0_bootstrap28.csv')
        i = features5_all.columns[1]
        features5_inds = features5_all[i]
        features5 = [(features5a[j], features5_inds.loc[j]) for j in features5_inds.index if features5_inds.loc[j] != 0]
        features5 = sorted(features5, key=itemgetter(1), reverse=True)
        with open('./data/comorbid_risk_features_wts_WBSLR_best.pickle', 'wb') as f:
            pickle.dump(features5, f)
        f.close()
        # tune the threshold for the proposed model on validation set
        valid_proba = pd.read_csv('./data/comorbid_risk_prediction_valid_20.csv')
        tune_proba_threshold_pred(valid_proba['pred'].values, valid_proba['response'], 1)

        # get example pt records
        data_dmckd[data_dmckd['ptid'] == '1658503'].to_csv('./data/comorbid_risk_pos_example1_withselectvar_obswindow.csv')  # rf predicted proba: 0.782
        data_dm_ckd3[data_dm_ckd3['ptid'] == '1658503'].to_csv('./data/comorbid_risk_pos_example1_obswindow.csv')
        data_dm_ckd[data_dm_ckd['ptid'] == '1658503'].to_csv('./data/comorbid_risk_pos_example1_all.csv')

        data_dmckd[data_dmckd['ptid'] == '651172'].to_csv('./data/comorbid_risk_pos_example2_withselectvar_obswindow.csv')  # rf predicted proba: 0.782
        data_dm_ckd3[data_dm_ckd3['ptid'] == '651172'].to_csv('./data/comorbid_risk_pos_example2_obswindow.csv')
        data_dm_ckd[data_dm_ckd['ptid'] == '651172'].to_csv('./data/comorbid_risk_pos_example2_all.csv')

        data_dm[data_dm['ptid'] == '1726071'].to_csv('./data/comorbid_risk_neg_example1_withselectvar_obswindow.csv')  # rf predicted proba: 0.782
        data_dm5[data_dm5['ptid'] == '1726071'].to_csv('./data/comorbid_risk_neg_example1_obswindow.csv')
        data_dm_ckd[data_dm_ckd['ptid'] == '1726071'].to_csv('./data/comorbid_risk_neg_example1_all.csv')

        data_dm[data_dm['ptid'] == '387115'].to_csv('./data/comorbid_risk_neg_example2_withselectvar_obswindow.csv')  # rf predicted proba: 0.782
        data_dm5[data_dm5['ptid'] == '387115'].to_csv('./data/comorbid_risk_neg_example2_obswindow.csv')
        data_dm2[data_dm2['ptid'] == '387115'].to_csv('./data/comorbid_risk_neg_example2_all.csv')


        # # analysis on example patient
        # import pandas as pd
        # exm1 = pd.read_csv('./data/comorbid_risk_neg_example2_all.csv')
        # exm1['adm_day'] = exm1['adm_date'].apply(lambda x: int(x / 60 / 24))
        # exm1['adm_month'] = exm1['adm_day'].apply(lambda x: int(x / 30))
        # del exm1['rank']
        # del exm1['dis_date']
        # del exm1['adm_date']
        # exm1 = exm1[['dxcat', 'adm_day', 'adm_month', 'first_dm_date']]

        # optimize the nonnegative weight logistic function
        import scipy as sp
        from scipy import optimize as opt
        import random

        def nnlr(X, y):
            """
            Non-negative Logistic Regression
            """

            def lr_cost(X, y, theta):
                m = len(y)
                return (1. / m) * (sp.dot(-y, sp.log(sigmoid(sp.dot(X, theta)))) - sp.dot((1 - y), sp.log(1 - sigmoid(sp.dot(X, theta)))))

            def lr_grad(X, y, theta):
                m = len(y)
                return (1. / m) * (sp.dot(X.T, sigmoid(sp.dot(X, theta)) - y))

            def sigmoid(z):
                return 1 / (1 + sp.exp(-z))

            N = X.shape[1]
            J = lambda theta: lr_cost(X, y, theta)
            J_grad = lambda theta: lr_grad(X, y, theta)
            random.seed(1)
            theta0 = np.random.uniform(0, 1, N)
            x, nfeval, rc = opt.fmin_tnc(J, theta0, fprime=J_grad, bounds=[(0, None)] * N,
                                         disp=0)
            return x

        n = 30
        preds_bagslr = pd.read_csv('./data/comorbid_risk_pred_y_' + str(n) + '.csv', index_col=0)
        y = preds_bagslr['response'].values
        del preds_bagslr['response']
        coefs = nnlr(preds_bagslr, y)
        pd.Series(coefs).to_csv('./data/comorbid_risk_bagging_weights_' + str(n) + '.csv')


