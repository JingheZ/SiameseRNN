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
        if 'gap_dm' in cols:
            df = df[['ptid', 'dxcat', 'gap_dm']].drop_duplicates()
            vals = [max(1, 12 - int((x / 24 / 60 - 180) / 30)) for x in df['gap_dm']]
            df['subw'] = [int((x - 1) / c) for x in vals]
        else:
            df = df[['ptid', 'dxcat', 'gap_copd']].drop_duplicates()
            vals = [min(int(-x / 24 / 60 / 30), 11) for x in df['gap_copd']]
            df['subw'] = [int(x / c) for x in vals]
    else:
        df.sort(['ptid', 'adm_date', 'dxcat'], ascending=[1, 1, 1], inplace=True)
        df = df[['ptid', 'dxcat', 'adm_date']].drop_duplicates()
    return df


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


if __name__ == '__main__':
    # ===================== load data =====================================
    with open('./data/data_copd_ptids.pickle', 'rb') as f:
        data_copd, ptids_copd = pickle.load(f)
    f.close()
    with open('./data/data_dm_ptids.pickle', 'rb') as f:
        data_dm, ptids_dm = pickle.load(f)
    f.close()

    data_copd_dm = pd.merge(data_copd, data_dm[['ptid', 'first_dm_date']].drop_duplicates(), how='inner', left_on='ptid',
                           right_on='ptid')
    data_copd_dm.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    data_copd_dm['gap_copd_dm'] = data_copd_dm['first_dm_date'] - data_copd_dm['first_copd_date']
    ptids_copd_dm = set(data_copd_dm['ptid'])  # 4803 pts
    d0 = data_copd_dm[data_copd_dm['first_dm_date'] >= 180 * 24 * 60]
    ptids_copd_dm0 = set(d0['ptid'])  # 1329 pts
    d1 = d0[d0['gap_copd_dm'] > (180 + 360) * 24 * 60].drop_duplicates()
    ptids_copd_dm1 = set(d1['ptid'])  # 561 pts

    # get the data as training of the dm class -1.5 to -0.5 years prior to first dm
    data_dm2 = data_dm[~data_dm['ptid'].isin(ptids_copd)]
    data_dm2 = data_dm2[~data_dm2['ptid'].isin(ptids_copd_dm)]
    data_dm3 = data_dm2[data_dm2['gap_dm'].between(180 * 24 * 60, 540 * 24 * 60)]
    ptids_dm3 = set(data_dm3['ptid'])  # 1065 pts

    # get the data for copd-dm (target group):
    data_copd_dm2 = data_copd_dm[data_copd_dm['ptid'].isin(ptids_copd_dm)]
    data_copd_dm2['gap_dm'] = data_copd_dm2['first_dm_date'] - data_copd_dm2['adm_date']
    data_copd_dm3 = data_copd_dm2[data_copd_dm2['gap_dm'].between(180 * 24 * 60, 540 * 24 * 60)]
    ptids_copd_dm3 = set(data_copd_dm3['ptid'])  # 1068 pts

    # get the copd data for training: 2.5 years of history after first copd: 1yr observation,
    # half year hold off, and 1yr prediction
    data_copd2 = data_copd[~data_copd['ptid'].isin(ptids_dm)]
    data_copd3 = data_copd2[data_copd2['gap_copd'] < -360 * 2.5 * 24 * 60]
    ptids_copd3 = set(data_copd3['ptid']) # 6259 pts
    data_copd4 = data_copd[data_copd['ptid'].isin(ptids_copd3)]
    data_copd5 = data_copd4[data_copd4['gap_copd'].between(-180 * 24 * 60, 0)]
    ptids_copd5 = set(data_copd5['ptid'])  # 6259 pts

    # get preliminary features
    counts_copd = get_counts_by_class(data_copd5, 0, len(ptids_copd5) * 0.05)
    counts_dm = get_counts_by_class(data_dm3, 1, len(ptids_dm3) * 0.05)
    counts_copddm = get_counts_by_class(data_copd_dm3, 1, len(ptids_copd_dm3) * 0.05)
    counts = counts_copd.append(counts_dm).append(counts_copddm).fillna(0)
    prelim_features = set(counts.columns[:-1]) # 64
    # update datasets to exclude unselected features
    data_copd = data_copd5[data_copd5['dxcat'].isin(prelim_features)]
    data_dm = data_dm3[data_dm3['dxcat'].isin(prelim_features)]
    data_copddm = data_copd_dm3[data_copd_dm3['dxcat'].isin(prelim_features)]
    ptids_copd = list(set(data_copd['ptid'].values.tolist())) # 6259
    ptids_dm = list(set(data_dm['ptid'].values.tolist()))  # 1022
    ptids_copddm = list(set(data_copddm['ptid'].values.tolist()))  # 1035
    # get aggregated counts
    counts_copd = get_counts_by_class(data_copd, 0, 0)
    counts_dm = get_counts_by_class(data_dm, 1, 0)
    counts_copddm = get_counts_by_class(data_copddm, 1, 0)
    counts = counts_copd.append(counts_dm).append(counts_copddm).fillna(0)
    counts.columns = ['cat' + i for i in counts.columns[:-1]] + ['response']
    counts.to_csv('./data/comorbid_task_counts.csv')

    # get subw counts
    counts_sub_copd = get_counts_subwindow(data_copd5, 0, prelim_features, 3)
    counts_sub_dm = get_counts_subwindow(data_dm3, 1, prelim_features, 3)
    counts_sub_copddm = get_counts_subwindow(data_copd_dm3, 1, prelim_features, 3)
    counts_sub = counts_sub_copd.append(counts_sub_dm).append(counts_sub_copddm).fillna(0)
    counts_sub.to_csv('./data/comorbid_task_counts_sub_by3momth.csv')

    counts_sub_copd = get_counts_subwindow(data_copd, 0, prelim_features, 2)
    counts_sub_dm = get_counts_subwindow(data_dm, 1, prelim_features, 2)
    counts_sub_copddm = get_counts_subwindow(data_copddm, 1, prelim_features, 2)
    counts_sub = counts_sub_copd.append(counts_sub_dm).append(counts_sub_copddm).fillna(0)
    counts_sub.to_csv('./data/comorbid_task_counts_sub_by2momth.csv')

    # ================ split train and testing data ========================================
    random.seed(5)
    ratio = 4
    # randomly select 60% for training and 40% for testing from target group
    train_ids_copd_dm, test_ids_copd_dm = split_target_data(np.array(ptids_copddm), 0.4)
    test_ids_copd = random.sample(ptids_copd, len(test_ids_copd_dm) * ratio)
    rest_copd_ptids = list(set(ptids_copd).difference(set(test_ids_copd)))
    test_ids = list(test_ids_copd) + list(test_ids_copd_dm)
    # # randomly select twice amount of target group from only copd group
    # train_ids_dm = ptids_dm
    # test_ids_copd = random.sample(ptids_copd, len(test_ids_copd_dm) * ratio)
    # rest_copd_ptids = list(set(ptids_copd).difference(set(test_ids_copd)))
    # train_ids_copd_a = random.sample(rest_copd_ptids, min((len(train_ids_copd_dm) + len(ptids_dm)) * ratio, len(rest_copd_ptids)))
    # train_ids_copd_b = random.sample(rest_copd_ptids, min(len(train_ids_copd_dm) * ratio, len(rest_copd_ptids)))
    # # testing ids
    # test_ids = list(test_ids_copd_dm) + test_ids_copd # 933 pts
    # # Training set 1: create training, including the pure dm patients
    # train_ids_a = list(train_ids_copd_dm) + train_ids_copd_a + train_ids_dm # 5238 pts
    # # Training set 2: create training, without pure dm patients
    # train_ids_b = list(train_ids_copd_dm) + train_ids_copd_b # 2172 pts

    # training data: train_ids
    train_ids_copd = rest_copd_ptids
    train_ids = train_ids_copd + list(train_ids_copd_dm)
    for r in np.arange(0, 5.1, 0.5):
        num_dm = r * len(train_ids_copd_dm)
        train_ids_dm = random.sample(ptids_dm, num_dm)
        train_ids += list(train_ids_dm)

        # =============== modeling =============================================================
        # baseline 1: aggregated count vector
        counts_x = counts[counts.columns[:-1]]
        counts_y = counts['response']
        features0 = counts.columns.tolist()[:-1]
        train_x0, train_y0, test_x0, test_y0 = split_shuffle_train_test_sets(train_ids, test_ids, counts_x, counts_y)

        clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
                                                                                         test_y0, features0, [100, 15, 2, 'rf'])
        # clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
        #                                                                                  test_y0, features0, [0.1, 15, 2, 'lr'])

        # baseline 2: subw count vector
        counts_sub_x = counts_sub[counts_sub.columns[1:]]
        del counts_sub_x['t0_cat127']
        counts_sub_y = counts_sub['response']
        features1 = counts_sub_x.columns.tolist()
        train_x1, train_y1, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids, test_ids, counts_sub_x, counts_sub_y)

        clf1, features_wts1, results_by_f1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features1,
                                                                        [100, 15, 2, 'rf'])
        # clf1, features_wts1, results_by_f1 = make_prediction_and_tuning(train_x1, train_y1, test_x1,
        #                                                                                  test_y1, features1, [0.01, 15, 1, 'lr'])


    # test_proba0a = make_predictions(train_x0, train_y0, test_x0, [1000, 15, 'rf'])
    # test_proba0b = make_predictions(train_x0, train_y0, test_x0, [1000, 15, 'lr'])
    # test_proba0c = make_predictions(train_x0, train_y0, test_x0, [0.01, 15, 'lr'])
