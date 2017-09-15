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


def make_prediction(train_x, train_y, test_x, test_y, s, param):
    clf = None
    if s == 'svm':
        clf = SVC(kernel=param[0], class_weight='balanced')
        # clf = SVC(kernel=param[0])
    elif s == 'rf':
        clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', class_weight='balanced')
        # clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy')
    elif s == 'lda':
        clf = LinearDiscriminantAnalysis()
    elif s == 'knn':
        clf = neighbors.KNeighborsClassifier(param[0], weights='distance')
    # elif s == 'xgb':
    #     param_dist = {'objective': 'binary:logistic', 'n_estimators': param[0], 'learning_rate': param[1]}
    #     xgb.XGBClassifier(**param_dist)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    result = metrics.classification_report(test_y, pred)
    auc = metrics.roc_auc_score(test_y, pred)
    return pred, result, auc


def split_shuffle_train_test_sets(train_ids, test_ids, X, y):
    train_x, test_x = X.ix[train_ids], X.ix[test_ids]
    train_y, test_y = y.ix[train_ids], y.ix[test_ids]
    train_x, train_y = shuffle(train_x, train_y, random_state=5)
    return train_x, train_y, test_x, test_y


def tune_proba_threshold_pred(pred_proba, y, test_pred_proba, test_y, b):
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
    pred = [1 if p > threshold else 0 for p in test_pred_proba]
    perfm = metrics.classification_report(test_y, pred)
    auc = metrics.roc_auc_score(test_y, pred)
    return threshold, perfm, auc, pred


def make_prediction_and_tuning(train_x, train_y, test_x, test_y, features, param):
    if param[3] == 'rf':
        clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=param[1], random_state=0)
        # clf = LogisticRegression(penalty='l1', C=param[0], n_jobs=param[1], random_state=0)
        clf.fit(train_x, train_y)
        pred_train = clf.predict_proba(train_x)
        pred_test = clf.predict_proba(test_x)
        train_pred_proba = [i[1] for i in pred_train]
        test_pred_proba = [i[1] for i in pred_test]
        # threshold tuning with f measure
        threshold_f, perfm_f, auc_f, pred_f = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba, test_y, param[2])
        print('Threshold %.3f tuned with f measure, AUC: %.3f' % (threshold_f, auc_f))
        print(perfm_f)
        # # threshold tuning with auc
        # threshold_a, perfm_a, auc_a, pred_a = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba, test_y, 'auc')
        # print('Threshold %.3f tuned with AUC, AUC: %.3f' % (threshold_a, auc_a))
        # print(perfm_a)
        # get the list of feature importance
        wts = clf.feature_importances_
        fts_wts = list(zip(features, wts))
        fts_wts_sorted = sorted(fts_wts, key=itemgetter(1), reverse=True)
        fts_wts = fts_wts_sorted
    else:
        clf = LogisticRegression(penalty='l1', C=param[0], n_jobs=param[1], random_state=0)
        clf.fit(train_x, train_y)
        pred_train = clf.predict_proba(train_x)
        pred_test = clf.predict_proba(test_x)
        train_pred_proba = [i[1] for i in pred_train]
        test_pred_proba = [i[1] for i in pred_test]
        # threshold tuning with f measure
        threshold_f, perfm_f, auc_f, pred_f = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba,
                                                                        test_y, param[2])
        print('Threshold %.3f tuned with f measure, AUC: %.3f' % (threshold_f, auc_f))
        print(perfm_f)
        # threshold tuning with auc
        # threshold_a, perfm_a, auc_a, pred_a = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba,
        #                                                                 test_y, 'auc')
        # print('Threshold %.3f tuned with AUC, AUC: %.3f' % (threshold_a, auc_a))
        # print(perfm_a)
        # get the list of feature importance
        # wts = clf.feature_importances_
        # fts_wts = list(zip(features, wts))
        # fts_wts_sorted = sorted(fts_wts, key=itemgetter(1), reverse=True)
        fts_wts = clf.coef_
    return clf, fts_wts, [threshold_f, pred_f]


def make_predictions(train_x, train_y, test_x, param):
    # train_x = train_x.as_matrix().astype(np.float)
    # test_x = test_x.as_matrix().astype(np.float)
    # train_y = train_y.as_matrix().astype(np.float)
    if param[2] == 'rf':
        clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=param[1], random_state=0)
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


def get_seq_item_counts(seq_dm, seq_control, seq_extra, cooccur_list, mvisit_list):
    # get items occurred at the same time
    def get_count_one_itemset(seq, c1, c2):
        ct1 = [1 if c1 in it and c2 in it else 0 for it in seq['dxcat'].values.tolist()]
        seq['cat' + str(c1) + '_' + str(c2)] = ct1
        count1 = seq[['ptid', 'cat' + str(c1) + '_' + str(c2)]].groupby('ptid').sum()
        return count1
    seq = pd.concat([seq_dm[['ptid', 'dxcat']], seq_control[['ptid', 'dxcat']], seq_extra[['ptid', 'dxcat']]], axis=0)
    count_ab = get_count_one_itemset(seq, cooccur_list[0][0], cooccur_list[0][1])
    for a, b in cooccur_list[1:]:
        countb = get_count_one_itemset(seq, a, b)
        count_ab = pd.concat([count_ab, countb], axis=1)

    # get same item occurred in different visits
    def get_count_two_visits(seq, c):
        ct1 = [1 if c in it else 0 for it in seq['dxcat'].values.tolist()]
        seq['var'] = ct1
        seq = seq[seq['var'] > 0]
        count1 = seq[['ptid', 'var']].groupby('ptid').count()
        count1['cat' + str(c) + 'to' + str(c)] = [i * (i - 1) * 0.5 for i in count1['var'].values]
        del count1['var']
        return count1
    count_cd = get_count_two_visits(seq, mvisit_list[0])
    for c in mvisit_list[1:]:
        countc = get_count_two_visits(seq, c)
        count_cd = pd.concat([count_cd, countc], axis=1)
    count_abcd = pd.concat([count_ab, count_cd], axis=1)
    return count_abcd


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

    # counts_sub_dm = get_counts_subwindow(data_dm5, 0, prelim_features, 3)
    # counts_sub_ckd = get_counts_subwindow(data_ckd3, 1, prelim_features, 3)
    # counts_sub_dmckd = get_counts_subwindow(data_dm_ckd3, 1, prelim_features, 3)
    # counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    # counts_sub.to_csv('./data/comorbid_task_counts_sub_by3momth.csv')
    #
    # counts_sub_dm = get_counts_subwindow(data_dm, 0, prelim_features, 2)
    # counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 2)
    # counts_sub_dmckd = get_counts_subwindow(data_dmckd, 1, prelim_features, 2)
    # counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    # counts_sub.to_csv('./data/comorbid_task_counts_sub_by2momth.csv')

    # =============== primary representation =============================================================
    # baseline 1: aggregated count vector
    counts_x = counts[counts.columns[:-1]]
    counts_y = counts['response']
    features1 = counts_x.columns.tolist()

    # baseline 2: subw count vector
    counts_sub_x = counts_sub[counts_sub.columns[1:]]
    counts_sub_y = counts_sub['response']
    features2 = counts_sub_x.columns.tolist()

    # # baseline 3: mining sequence patterns
    # # get the sequence by sub-windows
    # seq_dm = create_sequence(data_dm, 'comorbid_risk_dm')
    # seq_ckd = create_sequence(data_ckd, 'comorbid_risk_ckd')
    # seq_dmckd = create_sequence(data_dmckd, 'comorbid_risk_dmckd')
    # cooccur_list = [[258, 259], [53, 98], [204, 211]]
    # mvisit_list = [259, 211]
    # counts_bpsb = get_seq_item_counts(seq_dm, seq_dmckd, seq_ckd, cooccur_list, mvisit_list)
    # counts_bps = pd.concat([counts_bpsb, counts], axis=1).fillna(0)
    # counts_bps.to_csv('./data/counts_bps.csv')
    # counts_bps_y = counts_bps['response']
    # counts_bps_x = counts_bps
    # del counts_bps_x['response']
    # features3 = counts_bps_x.columns.tolist()
    #
    # # baseline 4: transitions
    # counts_trans = get_transition_counts(data_ckd, data_dm, data_dmckd, prelim_features)
    # counts_trans = pd.concat([counts_trans, counts_y], axis=1)
    # counts_trans_x = counts_trans[counts_trans.columns[:-1]]
    # counts_trans_y = counts_trans['response']
    # features4 = counts_trans_x.columns.tolist()

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
        #
        #
        train_x0, train_y0, test_x0, test_y0 = split_shuffle_train_test_sets(train_ids, test_ids, counts_x, counts_y)
        # # #
        train_x1, train_y1, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids, test_ids, counts_sub_x, counts_sub_y)
        # # # # #
        # # # # # # train_x4, train_y4, test_x4, test_y4 = split_shuffle_train_test_sets(train_ids, test_ids, counts_trans_x,
        # # # # # #                                                              counts_trans_y)
        clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
                                                                                         test_y0, features1, [100, 15, 1, 'rf'])
        # # # # # clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
        # # # # #                                                                                   test_y0, features0, [0.1, 15, 2, 'lr'])
        clf1, features_wts1, results_by_f1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features2,
                                                                        [100, 15, 2, 'rf'])
        # # # #
        # # # #


        features5_all = pd.read_csv('./data/sgl_coefs_alpha7_r3_bootstrap31.csv')
        i = features5_all.columns[1]
        features5_inds = features5_all[i]
        features5 = [features2[j] for j in features5_inds.index if features5_inds.loc[j] != 0]
        counts_sgl_x = counts_sub[features5]
        counts_sgl_y = counts_sub['response']
        train_x3, train_y3, test_x3, test_y3 = split_shuffle_train_test_sets(train_ids, test_ids, counts_sgl_x, counts_sgl_y)

        test_proba0a = make_predictions(train_x0, train_y0, test_x0, [30, 15, 'rf'])
        test_proba0b = make_predictions(train_x0, train_y0, test_x0, [200, 15, 'lr'])
        test_proba0c = make_predictions(train_x0, train_y0, test_x0, [0.05, 15, 'lr'])
        # #
        test_proba1a = make_predictions(train_x1, train_y1, test_x1, [30, 15, 'rf'])
        test_proba1b = make_predictions(train_x1, train_y1, test_x1, [200, 15, 'lr'])
        test_proba1c = make_predictions(train_x1, train_y1, test_x1, [0.05, 15, 'lr'])
        # # # #
        # # # # # test_proba2a = make_predictions(train_x2, train_y2, test_x2, [1000, 15, 'rf'])
        # # # # # test_proba2b = make_predictions(train_x2, train_y2, test_x2, [1000, 15, 'lr'])
        # # # # # test_proba2c = make_predictions(train_x2, train_y2, test_x2, [0.01, 15, 'lr'])
        # # # # #
        # # # # # test_proba3a = make_predictions(train_x3, train_y3, test_x3, [1000, 15, 'rf'])
        # # # # # test_proba3b = make_predictions(train_x3, train_y3, test_x3, [1000, 15, 'lr'])
        # # # # # test_proba3c = make_predictions(train_x3, train_y3, test_x3, [0.01, 15, 'lr'])
        # # # #
        test_proba4a = make_predictions(train_x3, train_y3, test_x3, [30, 15, 'rf'])
        test_proba4b = make_predictions(train_x3, train_y3, test_x3, [200, 15, 'lr'])
        test_proba4c = make_predictions(train_x3, train_y3, test_x3, [0.05, 15, 'lr'])
        # # # #
        test_proba = pd.DataFrame([test_proba0a, test_proba0b, test_proba0c, test_y0.values.tolist(),
                                   test_proba1a, test_proba1b, test_proba1c, test_y1.values.tolist(),
                                   test_proba4a, test_proba4b, test_proba4c, test_y3.values.tolist()])
        test_proba = test_proba.transpose()
        test_proba.columns = ['b1_rf', 'b1_lr', 'b1_lasso', 'b1_response', 'b2_rf', 'b2_lr', 'b2_lasso', 'b2_response',
                              'b5_rf', 'b5_lr', 'b5_lasso', 'b5_response']
        test_proba.to_csv('./data/comorbid_risk_test_proba_baselines_r' + str(r) + '.csv', index=False)
        # # # #
        # # # data_dm4[data_dm4['ptid'] == '769052'].to_csv('./data/example_dmpt.csv') # rf predicted proba: 0.782
        # # # data_control4[data_control4['ptid'] =='1819093'].to_csv('./data/example_controlpt.csv') # rf predicted proba: 0.033

        # for p in range(200):
        #     np.random.seed(p)
        #     sp = np.random.choice(train_ids, size=int(0.7 * len(train_ids)), replace=True)
        #     pd.Series(sp).to_csv('./data/comorbid_risk_train_ids_r' + str(r) + '_bootstrap' + str(p) + '.csv', index=False)
        #


