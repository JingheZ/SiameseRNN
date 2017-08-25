__author__ = 'jinghe'

'''
select the patients data according to the patient ids from both med and procedure datasets
'''

"""import packages"""
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
# import xgboost as xgb
import random
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils import shuffle
from operator import itemgetter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def getCode(element, CCS_dict):
    element = str(element)
    element = element.replace(".", "")
    element = element.capitalize()
    if CCS_dict.__contains__(element):
        dx = CCS_dict[element]
    elif len(element) < 5:
        element = element + '0'
        if CCS_dict.__contains__(element):
            dx = CCS_dict[element]
        else:
            element = element + '0'
            if CCS_dict.__contains__(element):
                dx = CCS_dict[element]
            else:
                dx = '0000 Others'
    elif not element[:-1].isdigit():
        element = element[:-1]
        if CCS_dict.__contains__(element):
            dx = CCS_dict[element]
        else:
            element = element + '0'
            if CCS_dict.__contains__(element):
                dx = CCS_dict[element]
            else:
                dx = '0000 Others'
    else:
        dx = '0000 Others'
    return dx


def dx2dxcat():
    filename1 = './data/dxref.csv'
    filename2 = './data/ccs_dx_icd10cm_2016.csv'
    def CCS(filename):
        data = pd.read_csv(filename, dtype=object)
        cols = data.columns
        data = data[cols[:3]]
        data.columns = ['icd', 'category', 'name']
        data['icd'] = data['icd'].str.replace("'", "")
        data['category'] = data['category'].str.replace("'", "")
        data['name'] = data['name'].str.replace("'", "")
        data['icd'] = data['icd'].str.replace(" ", "")
        data['category'] = data['category'].str.replace(" ", "")
        return data
    dxgrps9 = CCS(filename1)
    dxgrps10 = CCS(filename2)
    dxgrps = pd.concat([dxgrps9, dxgrps10], axis=0)
    dxgrps.index = dxgrps['icd']
    dxgrps_dict = dxgrps[['category']].to_dict()['category']
    del dxgrps_dict['']
    dxgrps_dict['F431'] = '651'
    icd10_init = ['R', 'L', 'M', 'G', 'W', 'S', 'V', 'F', 'D', 'X', 'P', 'T', 'N', 'O', 'Z', 'Y', 'I', 'C', 'Q', 'H', 'J', 'E', 'K']
    dxgrps_dict2 = {}
    for k, v in dxgrps_dict.items():
        if k[0] in icd10_init and len(k) > 5:
            if not dxgrps_dict2.__contains__(k[:5]):
                dxgrps_dict2[k[:5]] = v
    return dxgrps, dxgrps_dict, dxgrps_dict2


def process_dxs(data, dxgrps_dict, dxgrps_dict2):
    data['dx'] = data['itemid'].str.replace('.', '')
    dxs = data['dx'].values.tolist()
    dxcats = []
    for i in dxs:
        if dxgrps_dict.__contains__(i):
            dxcats.append(dxgrps_dict[i])
        elif dxgrps_dict2.__contains__(i):
            dxcats.append(dxgrps_dict2[i])
        else:
            dxcats.append('0')
    data['dxcat'] = dxcats
    data = data[data['dxcat'] != '0']
    return data


def merge_visit_and_dx(data_dx, visits):
    data = pd.merge(data_dx, visits, how='inner', left_on='vid', right_on='vid')
    data = data[['ptid_x', 'vid', 'pdx', 'dxcat', 'adm_date', 'dis_date', 'rank']]
    data.columns = ['ptid', 'vid', 'pdx', 'dxcat', 'adm_date', 'dis_date', 'rank']
    data.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    return data


def find_visit_gaps(data, dxcats):
    dms = data[data['dxcat'].isin(dxcats)]
    dm_ptids = set(dms['ptid'])
    print('%i patients' % len(set(dms['ptid'])))
    first_dm = dms[['ptid', 'adm_date']].drop_duplicates().groupby('ptid').min()
    first_dm.reset_index(inplace=True)
    first_dm.columns = ['ptid', 'first_dm_date']
    data_v2 = pd.merge(data, first_dm, how='inner', left_on='ptid', right_on='ptid')
    data_v2['gap_dm'] = data_v2['first_dm_date'] - data_v2['adm_date']
    return data_v2, dm_ptids


def find_patient_counts(data):
    x1 = data[data['first_dm_date'] >= 90 * 24 * 60]
    print('Number of patients with first DM after 90 days % i:')
    print(len(set(x1['ptid'])))
    x2 = data[data['first_dm_date'] >= 180 * 24 * 60]
    print('Number of patients with first DM after 180 days % i:')
    print(len(set(x2['ptid'])))
    x3 = data[data['gap_dm'].between(90 * 24 * 60, 455 * 24 * 60)]
    print('Number of patients with first DM between 90 and 455 days % i:')
    print(len(set(x3['ptid'])))
    x4 = data[data['gap_dm'].between(180 * 24 * 60, 635 * 24 * 60)]
    print('Number of patients with first DM between 180 and 455 days % i:')
    print(len(set(x4['ptid'])))
    return list(set(x1['ptid']))


def find_visit_gaps_control(data, target_ids, thres):
    # select patients with the records of the first visit
    pts_first_visit = data[data['adm_date'] == 0]
    ptids = set(pts_first_visit['ptid'])
    data = data[data['ptid'].isin(ptids)]
    # select patients visits of at least of threshold lengths of time
    pts_later_visits = data[data['adm_date'] > thres]
    ptids = set(pts_later_visits['ptid'])
    data = data[data['ptid'].isin(ptids)]
    # remove patients with the target dx
    data = data[~data['ptid'].isin(target_ids)]
    print('%i patients' % len(set(data['ptid'])))
    return data


def feature_selection_prelim(data, k):
    cols = data.columns
    X = data[cols[:-1]]
    y = data['response']
    feature_model = SelectKBest(chi2, k=k)
    X_new = feature_model.fit_transform(X, y)
    features = np.array(cols)[feature_model.get_support()]
    # y = np.array(y.tolist())
    X_new = X[features]
    return X_new, y, features


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
    clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=param[1], random_state=0, class_weight='balanced')
    clf.fit(train_x, train_y)
    pred_train = clf.predict_proba(train_x)
    pred_test = clf.predict_proba(test_x)
    train_pred_proba = [i[1] for i in pred_train]
    test_pred_proba = [i[1] for i in pred_test]
    # threshold tuning with f measure
    threshold_f, perfm_f, auc_f, pred_f = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba, test_y, param[2])
    print('Threshold %.3f tuned with f measure, AUC: %.3f' % (threshold_f, auc_f))
    print(perfm_f)
    # threshold tuning with auc
    threshold_a, perfm_a, auc_a, pred_a = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba, test_y, 'auc')
    print('Threshold %.3f tuned with AUC, AUC: %.3f' % (threshold_a, auc_a))
    print(perfm_a)
    # get the list of feature importance
    wts = clf.feature_importances_
    fts_wts = list(zip(features, wts))
    fts_wts_sorted = sorted(fts_wts, key=itemgetter(1), reverse=True)
    return clf, fts_wts_sorted, [threshold_f, pred_f], [threshold_a, pred_a]


def create_train_validate_test_sets_positive(X):
    rs = ShuffleSplit(n_splits=1, train_size=0.7, test_size=.3,
                      random_state=0)
    train_index = list(list(rs.split(X))[0][0])
    test_index = list(list(rs.split(X))[0][1])
    train_ids = list(X[train_index])
    test_ids = list(X[test_index])
    with open('./data/train_test_ptids_positive.pickle', 'wb') as f:
        pickle.dump([train_ids, test_ids], f)
    return train_ids, test_ids


def create_train_validate_test_sets_negative(X, size, test_ratio, train_ratio=1):
    random.shuffle(X)
    train_ids = random.sample(X, int(size * 0.7 * train_ratio))
    X2 = list(set(X).difference(set(train_ids)))
    test_ids = random.sample(X2, int(size * 0.3 * test_ratio))
    with open('./data/train_test_ptids_negative.pickle', 'wb') as f:
        pickle.dump([train_ids, test_ids], f)
    return train_ids, test_ids


def create_experiment_data(X, y, train_ids, test_ids):
    train_x = X.ix[train_ids]
    train_y = y.ix[train_ids]
    train_x, train_y = shuffle(train_x, train_y)
    test_x = X.ix[test_ids]
    test_y = y.ix[test_ids]

    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    test_y = test_y.values
    return train_x, train_y, test_x, test_y


def experiments(train_x, train_y, test_x, test_y):
    pred_svm, result_svm, auc_svm = make_prediction(train_x, train_y, test_x, test_y, 'svm', ['rbf'])
    print('SVM performance - AUC:%.3f' % auc_svm)
    print(result_svm)
    pred_lda, result_lda, auc_lda = make_prediction(train_x, train_y, test_x, test_y, 'lda', '')
    print('LDA performance - AUC:%.3f' % auc_lda)
    print(result_lda)
    pred_knn, result_knn, auc_knn = make_prediction(train_x, train_y, test_x, test_y, 'knn', [10])
    print('KNN performance - AUC:%.3f' % auc_knn)
    print(result_knn)
    pred_rf, result_rf, auc_rf = make_prediction(train_x, train_y, test_x, test_y, 'rf', [100])
    print('RF performance - AUC:%.3f' % auc_rf)
    print(result_rf)
    # pred_xgb, result_xgb, auc_xgb = make_prediction(train_x, train_y, test_x, test_y, 'lda', [100, 0.1])
    # print('XGB performance - AUC:%.3f' % auc_xgb)
    # print(result_xgb)


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
        if 'gap_dm' in cols:
            df = df[['ptid', 'vid', 'dxcat', 'gap_dm']].drop_duplicates()
            vals = [max(1, 18 - int((x / 24 / 60 - 90) / 30)) for x in df['gap_dm']]
            df['subw'] = [int((x - 1) / c) for x in vals]
        else:
            df = df[['ptid', 'dxcat', 'adm_date']].drop_duplicates()
            vals = [min(int(x / 24 / 60 / 30), 17) for x in df['adm_date']]
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
    for j in range(1, max(0, int(18/c))):
        cts = get_counts_one_window(counts0, j)
        dt = pd.merge(dt, cts, on='ptid', how='outer')
    dt['response'] = y
    dt.index = dt['ptid']
    del dt['ptid']
    dt.fillna(0, inplace=True)
    return dt


def split_train_test_ptids(y, test_sz=0.2):
    ptids = np.array(y.index.tolist())
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_sz, random_state=1)
    train_ids = []
    test_ids = []
    for train_index, test_index in sss.split(ptids, np.array(y)):
        train_ids, test_ids = ptids[train_index], ptids[test_index]
    return train_ids, test_ids


def split_train_test_sets(train_ids, test_ids, X, y):
    train_x, test_x = X.ix[train_ids], X.ix[test_ids]
    train_y, test_y = y.ix[train_ids], y.ix[test_ids]
    return train_x, train_y, test_x, test_y


def create_sequence(df, ptids, s):
    df = create_subwindows(df, 0)
    df = df[df['ptid'].isin(ptids)]
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


def get_seq_item_counts(seq_dm, seq_control, cooccur_list, mvisit_list):
    # get items occurred at the same time
    def get_count_one_itemset(seq, c1, c2):
        ct1 = [1 if c1 in it and c2 in it else 0 for it in seq['dxcat'].values.tolist()]
        seq['cat' + str(c1) + '_' + str(c2)] = ct1
        count1 = seq[['ptid', 'cat' + str(c1) + '_' + str(c2)]].groupby('ptid').sum()
        return count1
    seq = pd.concat([seq_dm[['ptid', 'dxcat']], seq_control[['ptid', 'dxcat']]], axis=0)
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


def tsne(data):
    model = TSNE(n_components=2, random_state=0)
    data2 = model.fit_transform(data)
    return data2


def viz_tsne(data, response):
    df = pd.DataFrame(data)
    df.columns = ['x', 'y']
    df['z'] = response
    sns.lmplot('x', 'y', data=df, fit_reg=False, hue="z", scatter_kws={"marker": "o", "s": 10})
    plt.title('Scatter Plot of Patient Populations')
    plt.xlabel('x')
    plt.ylabel('y')
    return df


if __name__ == '__main__':
    # ============================ DX Data =================================================
    with open('./data/visits_v4.pickle', 'rb') as f:
        visits = pickle.load(f)
    f.close()
    visits.columns = ['ptid', 'vid', 'IPorOP', 'adm_date', 'dis_date', 'rank']
    with open('./data/dxs_data_v2.pickle', 'rb') as f:
        data_dx = pickle.load(f)
    f.close()

    dxgrps, dxgrps_dict, dxgrps_dict2 = dx2dxcat()
    data_dx2 = process_dxs(data_dx, dxgrps_dict, dxgrps_dict2)
    data_dx2.head()

    data = merge_visit_and_dx(data_dx2, visits)
    # find patients with diabetes: dxcat = '49' or '50'
    data_dm, ptids_dm = find_visit_gaps(data, ['49', '50'])
    ptids_dm2 = find_patient_counts(data_dm)
    data_dm2 = data_dm[data_dm['ptid'].isin(ptids_dm2)]
    # get the visits in the observation window of the target patients
    data_dm3 = data_dm2[data_dm2['gap_dm'].between(180 * 24 * 60, 730 * 24 * 60)]
    ptids_dm3 = set(data_dm3['ptid']) # 5041 pts
    # # find patients with CHF: dxcat = '108'
    # data_chf, ptids_chf = find_visit_gaps(data, ['108'])
    # ptids_chf2 = find_patient_counts(data_chf)
    #
    # # find patients with CKD: dxcat = '158'
    # data_ckd = find_visit_gaps(data, ['158'])
    # find_patient_counts(data_ckd)
    #
    # # find patients with CKD: dxcat = '127'
    # data_copd = find_visit_gaps(data, ['127'])
    # find_patient_counts(data_copd)

    # find patients with at least four years of complete visits
    # 1. first visit date = 0
    # 2. one and a half year of observation window and three years of prediction window
    thres = 60 * 24 * 365 * 4
    data_control = find_visit_gaps_control(data, ptids_dm, thres)
    data_control2 = data_control[data_control['adm_date'] <= 24 * 60 * 545]
    data_control3 = data_control2[data_control2['dis_date'] <= 24 * 60 * 545]
    ptids_control = set(data_control3['ptid']) # 29752 pts

    # get the counts of dxcats of patients
    # counts_dm = get_counts_by_class(data_dm3, 1, 5664 * 0.05)
    counts_dm = get_counts_by_class(data_dm3, 1, 5041 * 0.05)
    counts_control = get_counts_by_class(data_control3, 0, 29752 * 0.05)
    counts = counts_dm.append(counts_control).fillna(0)
    prelim_features = set(counts.columns[:-1]) #40

    # filter out the rows with excluded features
    data_dm4 = data_dm3[data_dm3['dxcat'].isin(prelim_features)]
    data_control4 = data_control3[data_control3['dxcat'].isin(prelim_features)]
    counts_dm = get_counts_by_class(data_dm4, 1, 0)
    counts_control = get_counts_by_class(data_control4, 0, 0)
    counts = counts_dm.append(counts_control).fillna(0)

    counts.columns = ['cat' + i for i in counts.columns[:-1]] + ['response']
    counts.to_csv('./data/dm_control_counts.csv')
    # get training and testing ptids
    y = counts['response']
    train_ids, test_ids = split_train_test_ptids(y, 0.2)
    with open('./data/train_test_ptids.pickle', 'wb') as f:
        pickle.dump([train_ids, test_ids], f)
    f.close()
    pd.Series(train_ids).to_csv('./data/train_ids.csv', index=False)
    pd.Series(test_ids).to_csv('./data/test_ids.csv', index=False)
    # # get counts and do preliminary feature selection
    # counts_x, counts_y, features = feature_selection_prelim(counts, 50)
    # ============== Baseline 1: frequency =====================================
    # use actual ratio in training and testing:

    counts_x = counts[counts.columns[:-1]]
    counts_y = counts['response']
    features0 = counts.columns.tolist()[:-1]
    train_x0, train_y0, test_x0, test_y0 = split_train_test_sets(train_ids, test_ids, counts_x, counts_y)
    clf0, features_wts0, results_by_f0, results_by_auc0 = make_prediction_and_tuning(train_x0, train_y0, test_x0, test_y0, features0, [1000, 15, 5])

    # ============= baseline 2: frequency in sub-window ===================================
    # every season: get the counts and then append
    counts_sub_dm = get_counts_subwindow(data_dm3, 1, prelim_features, 3)
    counts_sub_control = get_counts_subwindow(data_control3, 0, prelim_features, 3)
    counts_sub = counts_sub_dm.append(counts_sub_control).fillna(0)
    counts_sub.to_csv('./data/counts_sub.csv')

    counts_sub_x = counts_sub[counts_sub.columns[:-1]]
    counts_sub_y = counts_sub['response']
    features1 = counts_sub.columns.tolist()[:-1]
    train_x1, train_y1, test_x1, test_y1 = split_train_test_sets(train_ids, test_ids, counts_sub_x, counts_sub_y)
    clf1, features_wts1, results_by_f1, results_by_auc1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features1, [1000, 15, 5])

    # ============== baseline 3: mining sequence patterns =============================================
    # get the sequence by sub-windows
    seq_dm = create_sequence(data_dm4, train_ids, 'dm_train')
    seq_control = create_sequence(data_control4, train_ids, 'control_train')
    # load the selected features using sequential pattern mining SPADE
    features2_dm = pd.read_csv('./data/result_dm.txt', delimiter=' -1', header=None)
    # features2_control = pd.read_csv('./data/result_control.txt', delimiter=' -1', header=None)
    # merge the features selected to get the selected single features
    features2a = features2_dm[0].values.tolist()[:-2] + ['167', '663']
    features2a_names = ['response'] + ['cat' + i for i in features2a]
    counts_bps = counts[features2a_names]

    cooccur_list = [[258, 259], [53, 98], [256, 259], [212, 259], [256, 258], [167, 258], [204, 211]]
    mvisit_list = [259, 133, 98, 258, 211, 205]
    counts_bpsb = get_seq_item_counts(seq_dm, seq_control, cooccur_list, mvisit_list)
    counts_bps = pd.concat([counts, counts_bpsb], axis=1).fillna(0)
    # To do: need to add 258 -> 167
    counts_bps_y = counts_bps['response']
    counts_bps_x = counts_bps
    del counts_bps_x['response']
    features2 = counts_bps_x.columns.tolist()
    train_x2, train_y2, test_x2, test_y2 = split_train_test_sets(train_ids, test_ids, counts_bps_x, counts_bps_y)
    clf2, features_wts2, results_by_f2, results_by_auc2 = make_prediction_and_tuning(train_x2, train_y2, test_x2, test_y2, features2, [1000, 15, 5])

    # ============= Proposed: frequency in sub-window and selected by sgl===================================
    features2_all = pd.read_csv('./data/SGL_coefs.csv')
    del features2_all['Unnamed: 0']
    data_cols = counts_sub.columns
    feature_names_sgl = []
    for i in features2_all.columns:
        print(i)
        features2_inds = features2_all[i]
        features2 = [data_cols[j] for j in features2_inds.index if features2_inds.loc[j] != 0]
        feature_names_sgl.append(features2)
        if len(features2) > 0:
            print('%i features are selected in SGL' % len(features2))
            counts_sgl_x = counts_sub[features2]
            counts_sgl_y = counts_sub['response']
            train_x2, train_y2, test_x2, test_y2 = split_train_test_sets(train_ids, test_ids, counts_sgl_x, counts_sgl_y)
            clf2, features_wts2, results_by_f2, results_by_auc2 = make_prediction_and_tuning(train_x2, train_y2, test_x2, test_y2, features2, [1000, 15, 2])
        else:
            print('No feature is selected in SGL!')



    # ============= Add t-sne for visualization ==========================================================
    output_train_tsne0 = viz_tsne(train_x0)
    output_test_tsne0 = viz_tsne(train_x0)

    output_train_tsne1 = viz_tsne(train_x1)
    output_test_tsne1 = viz_tsne(train_x1)

    output_train_tsne2 = viz_tsne(train_x2)
    output_test_tsne2 = viz_tsne(train_x2)

    # output_train_tsne3 = viz_tsne(train_x3)
    # output_test_tsne3 = viz_tsne(train_x3)