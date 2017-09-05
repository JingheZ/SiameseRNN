"""
Transfer learning for ckdid risk prediction
1. boosted or bagged SGL
2. significance: used domain adaptation for ckdid risk prediction; considers temporal info; works for small samples
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
            vals = [max(1, 6 - int((x / 24 / 60) / 30)) for x in df['gap_dm']]
            df['subw'] = [int((x - 1) / c) for x in vals]
        else:
            df = df[['ptid', 'dxcat', 'gap_ckd']].drop_duplicates()
            vals = [max(1, 6 - int((x / 24 / 60 - 0) / 30)) for x in df['gap_ckd']]
            df['subw'] = [int((x - 1) / c) for x in vals]
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
    for j in range(1, max(0, int(6/c))):
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


def create_train_validate_test_sets_positive(ptids, y):
    rs = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=.3,
                      random_state=0)
    rs.get_n_splits(ptids, y)
    train_index = list(list(rs.split(ptids, y))[0][0])
    test_index = list(list(rs.split(ptids, y))[0][1])
    train_ids = list(ptids[train_index])
    test_ids = list(ptids[test_index])
    # with open('./data/ckdid_risk_target_train_test_ptids.pickle', 'wb') as f:
    #     pickle.dump([train_ids, test_ids], f)
    return train_ids, test_ids


if __name__ == '__main__':
    # ===================== load data =====================================
    # with open('./data/data_all4ckdidity.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # f.close()
    random.seed(1)
    with open('./data/data_dm_ptids.pickle', 'rb') as f:
        data_dm, ptids_dm = pickle.load(f)
    f.close()
    with open('./data/data_ckd_ptids.pickle', 'rb') as f:
        data_ckd, ptids_ckd = pickle.load(f)
    f.close()

    # with open('./data/data_ckd_ptids.pickle', 'rb') as f:
    #     data_ckd, ptids_ckd = pickle.load(f)
    # f.close()
    # cohort:
    # data of 4 years, 1 year prior to DM, and 3 month hold-off, total 3 year for prediction of ckdidity
    data_dm_v2 = data_dm[data_dm['gap_dm'] <= -360 * 3 * 24 * 60]
    data_dm_v3 = data_dm[data_dm['gap_dm'] >= 180 * 24 * 60]
    data_dm_ptids = set(data_dm_v2['ptid'].values).intersection(set(data_dm_v3['ptid'].values)) # 1001

    data_dm_ckd = pd.merge(data_dm, data_ckd[['ptid', 'first_ckd_date']].drop_duplicates(), how='inner', left_on='ptid',
                           right_on='ptid')
    data_dm_ckd.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    # data_dm_ckd['gap_dm_ckd'] = data_dm_ckd['first_dm_date'] - data_dm_ckd['first_ckd_date']

    # ckd onset prior to prediction window
    data_dm_ckd_earlyckd = data_dm_ckd[data_dm_ckd['first_dm_date'] - data_dm_ckd['first_ckd_date'] > -90 * 24 * 60]
    # ckd onset after prediction window
    data_dm_ckd_lateckd = data_dm_ckd[data_dm_ckd['first_dm_date'] - data_dm_ckd['first_ckd_date'] < -360 * 3 * 24 * 60]
    #
    target_pos_ids = data_dm_ptids.intersection(set(data_dm_ckd['ptid'].values)).difference(set(data_dm_ckd_earlyckd['ptid'].values)).difference(set(data_dm_ckd_lateckd['ptid'].values))
    target_neg_ids = data_dm_ptids.difference(set(data_dm_ckd['ptid'].values))\
        .union(data_dm_ptids.intersection(set(data_dm_ckd_lateckd['ptid'].values)))

    # target_pos_ids = data_dm_ptids.intersection(set(data_dm_ckd['ptid'].values)).difference(set(data_dm_ckd_earlyckd['ptid'].values))
    # target_neg_ids = data_dm_ptids.difference(set(data_dm_ckd['ptid'].values))

    data_dm = data_dm[data_dm['gap_dm'].between(0, 180 * 24 * 60)]
    target_pos_data = data_dm[data_dm['ptid'].isin(target_pos_ids)]
    target_neg_data = data_dm[data_dm['ptid'].isin(target_neg_ids)]

    counts_target_pos = get_counts_by_class(target_pos_data, 1, len(target_pos_ids) * 0.05)
    counts_target_neg = get_counts_by_class(target_neg_data, 0, len(target_neg_ids) * 0.05)
    counts_target = counts_target_pos.append(counts_target_neg).fillna(0)
    prelim_features = set(counts_target.columns[:-1])  # 59

    target_pos_data = target_pos_data[target_pos_data['dxcat'].isin(prelim_features)]
    target_neg_data = target_neg_data[target_neg_data['dxcat'].isin(prelim_features)]
    target_pos_ids = list(set(target_pos_data['ptid'].values)) # 25
    target_neg_ids = list(set(target_neg_data['ptid'].values)) # 761

    data_ckd_v2 = data_ckd[data_ckd['gap_ckd'] >= 180 * 24 * 60]
    data_ckd_ptids = set(data_ckd_v2['ptid'].values).difference(set(ptids_dm))
    data_ckd = data_ckd[data_ckd['ptid'].isin(data_ckd_ptids)]
    # data_ckd = data_ckd[data_ckd['gap_ckd'].between(180 * 24 * 60, 360 * 24 * 60)]
    data_ckd = data_ckd[data_ckd['gap_ckd'].between(0, 180 * 24 * 60)]
    data_ckd = data_ckd[data_ckd['dxcat'].isin(prelim_features)]

    # get aggregated counts
    counts_target_pos = get_counts_by_class(target_pos_data, 1, 0)
    counts_target_neg = get_counts_by_class(target_neg_data, 0, 0)
    counts_ckd = get_counts_by_class(data_ckd, 1, 0)
    counts = counts_target_pos.append(counts_target_neg).append(counts_ckd).fillna(0)
    counts.columns = ['cat' + i for i in counts_target.columns[:-1]] + ['response']

    # get subw counts
    counts_sub_target_pos = get_counts_subwindow(target_pos_data, 1, prelim_features, 3)
    counts_sub_target_neg = get_counts_subwindow(target_neg_data, 0, prelim_features, 3)
    counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 3)
    counts_sub = counts_sub_target_pos.append(counts_sub_target_neg).append(counts_sub_ckd).fillna(0)

    # shufflesplit
    target_neg_ids_sample = random.sample(target_neg_ids, len(target_pos_ids) * 2)
    target_ids_sample = list(target_neg_ids_sample) + target_pos_ids
    ys = [0] * len(target_neg_ids_sample) + [1] * len(target_pos_ids)
    train_ids, test_ids = create_train_validate_test_sets_positive(np.array(target_ids_sample), ys)

    rest_dm_ptids = set(target_neg_ids).difference(target_neg_ids_sample)
    ckd_ids_sample = random.sample(list(set(data_ckd['ptid'].values)), len(rest_dm_ptids))
    train_ids_v2 = train_ids + list(rest_dm_ptids) + list(ckd_ids_sample)
    train_ids_v2 = train_ids + list(ckd_ids_sample)
    # ================================== Modeling ========================================
    # baseline 1: aggregated counts of target data only
    counts_x = counts[counts.columns[:-1]]
    counts_y = counts['response']
    features0 = counts_x.columns.tolist()
    train_x0, train_y0, test_x0, test_y0 = split_shuffle_train_test_sets(train_ids, test_ids, counts_x, counts_y)
    clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
                                                                    test_y0, features0, [100, 15, 1, 'rf'])
    # baseline 2: subw counts of target data only
    counts_sub_x = counts_sub[counts_sub.columns[1:]]
    counts_sub_y = counts_sub['response']
    features1 = counts_sub_x.columns.tolist()
    train_x1, train_y1, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids, test_ids, counts_sub_x,
                                                                         counts_sub_y)
    clf1, features_wts1, results_by_f1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features1,
                                                                    [100, 15, 1, 'rf'])

    # baseline 3: aggregated counts of target data and control data
    counts_x = counts[counts.columns[:-1]]
    counts_y = counts['response']
    features0 = counts_x.columns.tolist()
    train_x0, train_y0, test_x0, test_y0 = split_shuffle_train_test_sets(train_ids_v2, test_ids, counts_x, counts_y)
    clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
                                                                    test_y0, features0, [100, 15, 1, 'rf'])
    # baseline 4: subw counts of target data and control data
    counts_sub_x = counts_sub[counts_sub.columns[1:]]
    counts_sub_y = counts_sub['response']
    features1 = counts_sub_x.columns.tolist()
    train_x1, train_y1, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids_v2, test_ids, counts_sub_x,
                                                                         counts_sub_y)
    clf1, features_wts1, results_by_f1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features1,
                                                                    [100, 15, 1, 'rf'])

    # # get the data as training of the ckd class -1.5 to -0.5 years prior to first ckd
    # data_ckd2 = data_ckd[~data_ckd['ptid'].isin(ptids_dm)]
    # data_ckd3 = data_ckd2[data_ckd2['gap_ckd'].between(180 * 24 * 60, 540 * 24 * 60)]
    # ptids_ckd3 = set(data_ckd3['ptid'])  # 1065 pts

    # get preliminary features
    counts_dm = get_counts_by_class(data_dm5, 0, len(ptids_dm5) * 0.05)
    counts_ckd = get_counts_by_class(data_ckd3, 1, len(ptids_ckd3) * 0.05)
    counts_dmckd = get_counts_by_class(data_dm_ckd3, 1, len(ptids_dm_ckd3) * 0.05)
    counts = counts_dm.append(counts_ckd).append(counts_dmckd).fillna(0)
    prelim_features = set(counts.columns[:-1]) # 63
    # update datasets to exclude unselected features
    data_dm = data_dm5[data_dm5['dxcat'].isin(prelim_features)]
    data_ckd = data_ckd3[data_ckd3['dxcat'].isin(prelim_features)]
    data_dmckd = data_dm_ckd3[data_dm_ckd3['dxcat'].isin(prelim_features)]
    ptids_dm = list(set(data_dm['ptid'].values.tolist())) # 7436
    ptids_ckd = list(set(data_ckd['ptid'].values.tolist()))  # 1022
    ptids_dmckd = list(set(data_dmckd['ptid'].values.tolist()))  # 401
    # get aggregated counts
    counts_dm = get_counts_by_class(data_dm, 0, 0)
    counts_ckd = get_counts_by_class(data_ckd, 1, 0)
    counts_dmckd = get_counts_by_class(data_dmckd, 1, 0)
    counts = counts_dm.append(counts_ckd).append(counts_dmckd).fillna(0)
    counts.columns = ['cat' + i for i in counts.columns[:-1]] + ['response']
    counts.to_csv('./data/ckdid_task_counts.csv')

    # get subw counts
    counts_sub_dm = get_counts_subwindow(data_dm5, 0, prelim_features, 3)
    counts_sub_ckd = get_counts_subwindow(data_ckd3, 1, prelim_features, 3)
    counts_sub_dmckd = get_counts_subwindow(data_dm_ckd3, 1, prelim_features, 3)
    counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    counts_sub.to_csv('./data/ckdid_task_counts_sub_by3momth.csv')

    counts_sub_dm = get_counts_subwindow(data_dm, 0, prelim_features, 2)
    counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 2)
    counts_sub_dmckd = get_counts_subwindow(data_dmckd, 1, prelim_features, 2)
    counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    counts_sub.to_csv('./data/ckdid_task_counts_sub_by2momth.csv')

    # ================ split train and testing data ========================================
    random.seed(5)
    ratio = 2
    # randomly select 60% for training and 40% for testing from target group
    train_ids_dm_ckd, test_ids_dm_ckd = split_target_data(np.array(ptids_dmckd), 0.5)
    test_ids_dm = random.sample(ptids_dm, len(test_ids_dm_ckd) * ratio)
    rest_dm_ptids = list(set(ptids_dm).difference(set(test_ids_dm)))
    test_ids = list(test_ids_dm) + list(test_ids_dm_ckd)
    # # randomly select twice amount of target group from only dm group

    # train_ids_dm = rest_dm_ptids
    # train_ids = train_ids_ckd + list(train_ids_ckd_dm)
    train_ids = list(train_ids_dm_ckd)
    for r in np.arange(0, 2.6, 0.5):
        num_ckd = r * len(train_ids_dm_ckd)
        train_ids_ckd = random.sample(ptids_ckd, num_ckd)
        train_ids_dm = random.sample(rest_dm_ptids, (num_ckd + len(train_ids_dm_ckd)) * ratio)
        # train_ids_dm = random.sample(rest_dm_ptids, num_ckd * ratio)
        train_ids += list(train_ids_dm) + list(train_ids_ckd)

        # =============== modeling =============================================================
        # baseline 1: aggregated count vector
        counts_x = counts[counts.columns[:-1]]
        counts_y = counts['response']
        features0 = counts_x.columns.tolist()
        train_x0, train_y0, test_x0, test_y0 = split_shuffle_train_test_sets(train_ids, test_ids, counts_x, counts_y)

        clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
                                                                                         test_y0, features0, [100, 15, 2, 'rf'])
        # clf0, features_wts0, results_by_f0 = make_prediction_and_tuning(train_x0, train_y0, test_x0,
        #                                                                                  test_y0, features0, [0.1, 15, 2, 'lr'])

        # baseline 2: subw count vector
        counts_sub_x = counts_sub[counts_sub.columns[1:]]
        # del counts_sub_x['t0_cat127']
        counts_sub_y = counts_sub['response']
        features1 = counts_sub_x.columns.tolist()
        train_x1, train_y1, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids, test_ids, counts_sub_x, counts_sub_y)

        clf1, features_wts1, results_by_f1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features1,
                                                                        [100, 15, 2, 'rf'])

