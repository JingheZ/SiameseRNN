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
            ct = data[i].value_counts(dropna=False)[0]
            if num - ct > thres:
                cols_updated.append(i)
        data = data[cols_updated]
        return data
    df = df[['ptid', 'vid', 'dxcat']].drop_duplicates()
    counts = df[['ptid', 'dxcat']].groupby(['ptid', 'dxcat']).size().unstack('dxcat').fillna(0)
    counts = filter_rare_columns(counts, thres)
    counts['response'] = y
    return counts


def create_subwindows(df, c=1):
    cols = df.columns
    if c > 0:
        if 'gap_ckd' in cols:
            df = df[['ptid', 'dxcat', 'gap_ckd']].drop_duplicates()
            vals = [max(1, 12 - int((x / 24 / 60 - 180) / 30)) for x in df['gap_ckd']]
            df['subw'] = [int((x - 1) / c) for x in vals]
        else:
            df = df[['ptid', 'dxcat', 'gap_dm']].drop_duplicates()
            vals = [min(int(-x / 24 / 60 / 30), 11) for x in df['gap_dm']]
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
    train_x, train_y = shuffle(train_x, train_y, random_state=1)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    # ===================== load data =====================================
    with open('./data/data_dm_ptids.pickle', 'rb') as f:
        data_dm, ptids_dm = pickle.load(f)
    f.close()
    with open('./data/data_ckd_ptids.pickle', 'rb') as f:
        data_ckd, ptids_ckd = pickle.load(f)
    f.close()

    data_dm_ckd = pd.merge(data_dm, data_ckd[['ptid', 'first_ckd_date']].drop_duplicates(), how='inner', left_on='ptid',
                           right_on='ptid')
    data_dm_ckd.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    data_dm_ckd['gap_dm_ckd'] = data_dm_ckd['first_ckd_date'] - data_dm_ckd['first_dm_date']
    ptids_dm_ckd = set(data_dm_ckd['ptid'])  # 4803 pts
    d0 = data_dm_ckd[data_dm_ckd['first_ckd_date'] >= 180 * 24 * 60]
    ptids_dm_ckd0 = set(d0['ptid'])  # 1329 pts
    d1 = d0[d0['gap_dm_ckd'] > (180 + 360) * 24 * 60].drop_duplicates()
    ptids_dm_ckd1 = set(d1['ptid'])  # 561 pts

    # get the data as training of the CKD class -1.5 to -0.5 years prior to first CKD
    data_ckd2 = data_ckd[~data_ckd['ptid'].isin(ptids_dm)]
    data_ckd3 = data_ckd2[data_ckd2['gap_ckd'].between(180 * 24 * 60, 540 * 24 * 60)]
    ptids_ckd3 = set(data_ckd3['ptid'])  # 1065 pts

    # get the data for dm-ckd (target group):
    data_dm_ckd2 = data_dm_ckd[data_dm_ckd['ptid'].isin(ptids_dm_ckd)]
    data_dm_ckd2['gap_ckd'] = data_dm_ckd2['first_ckd_date'] - data_dm_ckd2['adm_date']
    data_dm_ckd3 = data_dm_ckd2[data_dm_ckd2['gap_ckd'].between(180 * 24 * 60, 540 * 24 * 60)]
    ptids_dm_ckd3 = set(data_dm_ckd3['ptid'])  # 1068 pts

    # get the dm data for training: 2.5 years of history after first dm: 1yr observation,
    # half year hold off, and 1yr prediction
    data_dm2 = data_dm[~data_dm['ptid'].isin(ptids_ckd)]
    data_dm3 = data_dm2[data_dm2['gap_dm'] < -360 * 2.5 * 24 * 60]
    ptids_dm3 = set(data_dm3['ptid']) # 6259 pts
    data_dm4 = data_dm[data_dm['ptid'].isin(ptids_dm3)]
    data_dm5 = data_dm4[data_dm4['gap_dm'].between(-360 * 24 * 60, 0)]
    ptids_dm5 = set(data_dm5['ptid'])  # 6259 pts

    # get preliminary features
    counts_dm = get_counts_by_class(data_dm5, 0, len(ptids_dm5) * 0.05)
    counts_ckd = get_counts_by_class(data_ckd3, 1, len(ptids_ckd3) * 0.05)
    counts_dmckd = get_counts_by_class(data_dm_ckd3, 1, len(ptids_dm_ckd3) * 0.05)
    counts = counts_dm.append(counts_ckd).append(counts_dmckd).fillna(0)
    prelim_features = set(counts.columns[:-1]) # 64
    # update datasets to exclude unselected features
    data_dm = data_dm5[data_dm5['dxcat'].isin(prelim_features)]
    data_ckd = data_ckd3[data_ckd3['dxcat'].isin(prelim_features)]
    data_dmckd = data_dm_ckd3[data_dm_ckd3['dxcat'].isin(prelim_features)]
    ptids_dm = list(set(data_dm['ptid'].values.tolist())) # 6259
    ptids_ckd = list(set(data_ckd['ptid'].values.tolist()))  # 1022
    ptids_dmckd = list(set(data_dmckd['ptid'].values.tolist()))  # 1035
    # get aggregated counts
    counts_dm = get_counts_by_class(data_dm, 0, 0)
    counts_ckd = get_counts_by_class(data_ckd, 1, 0)
    counts_dmckd = get_counts_by_class(data_dmckd, 1, 0)
    counts = counts_dm.append(counts_ckd).append(counts_dmckd).fillna(0)
    counts.columns = ['cat' + i for i in counts.columns[:-1]] + ['response']
    counts.to_csv('./data/comorbid_task_counts.csv')

    # get subw counts
    counts_sub_dm = get_counts_subwindow(data_dm, 0, prelim_features, 3)
    counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 3)
    counts_sub_dmckd = get_counts_subwindow(data_dmckd, 1, prelim_features, 3)
    counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    counts_sub.to_csv('./data/comorbid_task_counts_sub_by3momth.csv')

    counts_sub_dm = get_counts_subwindow(data_dm, 0, prelim_features, 2)
    counts_sub_ckd = get_counts_subwindow(data_ckd, 1, prelim_features, 2)
    counts_sub_dmckd = get_counts_subwindow(data_dmckd, 1, prelim_features, 2)
    counts_sub = counts_sub_dm.append(counts_sub_ckd).append(counts_sub_dmckd).fillna(0)
    counts_sub.to_csv('./data/comorbid_task_counts_sub_by2momth.csv')

    # ================ split train and testing data ========================================
    random.seed(1)
    # randomly select 60% for training and 30% for testing from target group
    train_ids_dm_ckd, test_ids_dm_ckd = split_target_data(np.array(ptids_dmckd), 0.3)
    # randomly select twice amount of target group from only dm group
    train_ids_ckd = ptids_ckd
    test_ids_dm = random.sample(ptids_dm, len(test_ids_dm_ckd) * 2)
    train_ids_dm_a = random.sample(list(set(ptids_dm).difference(set(test_ids_dm))),
                                  (len(train_ids_dm_ckd) + len(ptids_ckd)) * 2)
    train_ids_dm_b = random.sample(list(set(ptids_dm).difference(set(test_ids_dm))), len(train_ids_dm_ckd) * 2)
    # testing ids
    test_ids = list(test_ids_dm_ckd) + test_ids_dm # 933 pts
    # Training set 1: create training, including the pure CKD patients
    train_ids_a = list(train_ids_dm_ckd) + train_ids_dm_a + train_ids_ckd # 5238 pts
    # Training set 2: create training, without pure CKD patients
    train_ids_b = list(train_ids_dm_ckd) + train_ids_dm_b # 2172 pts

    # =============== modeling =============================================================
    # baseline 1: aggregated count vector
    counts_x = counts[counts.columns[:-1]]
    counts_y = counts['response']
    features0 = counts.columns.tolist()[:-1]
    train_x0a, train_y0a, test_x0, test_y0 = split_shuffle_train_test_sets(train_ids_a, test_ids, counts_x, counts_y)
    train_x0b, train_y0b, test_x0, test_y0 = split_shuffle_train_test_sets(train_ids_b, test_ids, counts_x, counts_y)

    # baseline 2: subw count vector
    counts_sub_x = counts_sub[counts_sub.columns[1:]]
    counts_sub_y = counts_sub['response']
    features1 = counts_sub.columns.tolist()[1:]
    train_x1a, train_y1a, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids_a, test_ids, counts_sub_x, counts_sub_y)
    train_x1b, train_y1b, test_x1, test_y1 = split_shuffle_train_test_sets(train_ids_b, test_ids, counts_sub_x, counts_sub_y)




    # test_proba0a = make_predictions(train_x0, train_y0, test_x0, [1000, 15, 'rf'])
    # test_proba0b = make_predictions(train_x0, train_y0, test_x0, [1000, 15, 'lr'])
    # test_proba0c = make_predictions(train_x0, train_y0, test_x0, [0.01, 15, 'lr'])
