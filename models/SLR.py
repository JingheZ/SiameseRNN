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
from sklearn.linear_model import LogisticRegression
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
from sklearn.decomposition import PCA


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
    data = data[['ptid_x', 'vid', 'pdx', 'dxcat', 'adm_date', 'dis_date', 'rank', 'dx']]
    data.columns = ['ptid', 'vid', 'pdx', 'dxcat', 'adm_date', 'dis_date', 'rank', 'dx']
    data.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    return data


def find_visit_gaps(data, dxcats, s):
    dms = data[data['dxcat'].isin(dxcats)]
    dm_ptids = set(dms['ptid'])
    print('%i patients' % len(set(dms['ptid'])))
    first_dm = dms[['ptid', 'adm_date']].drop_duplicates().groupby('ptid').min()
    first_dm.reset_index(inplace=True)
    first_dm.columns = ['ptid', 'first_' + s + '_date']
    data_v2 = pd.merge(data, first_dm, how='inner', left_on='ptid', right_on='ptid')
    data_v2['gap_' + s] = data_v2['first_' + s + '_date'] - data_v2['adm_date']
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
    if param[3] == 'rf':
        clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=param[1], random_state=0, class_weight='balanced')
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
        # threshold tuning with auc
        threshold_a, perfm_a, auc_a, pred_a = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba, test_y, 'auc')
        print('Threshold %.3f tuned with AUC, AUC: %.3f' % (threshold_a, auc_a))
        print(perfm_a)
        # get the list of feature importance
        # wts = clf.feature_importances_
        # fts_wts = list(zip(features, wts))
        # fts_wts_sorted = sorted(fts_wts, key=itemgetter(1), reverse=True)
        fts_wts = clf.feature_importances_
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
        threshold_a, perfm_a, auc_a, pred_a = tune_proba_threshold_pred(train_pred_proba, train_y, test_pred_proba,
                                                                        test_y, 'auc')
        print('Threshold %.3f tuned with AUC, AUC: %.3f' % (threshold_a, auc_a))
        print(perfm_a)
        # get the list of feature importance
        # wts = clf.feature_importances_
        # fts_wts = list(zip(features, wts))
        # fts_wts_sorted = sorted(fts_wts, key=itemgetter(1), reverse=True)
        fts_wts = clf.coef_
    return clf, fts_wts, [threshold_f, pred_f], [threshold_a, pred_a]


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
            vals = [max(1, 12 - int((x / 24 / 60 - 180) / 30)) for x in df['gap_dm']]
            df['subw'] = [int((x - 1) / c) for x in vals]
        else:
            df = df[['ptid', 'dxcat', 'adm_date']].drop_duplicates()
            vals = [min(int(x / 24 / 60 / 30), 11) for x in df['adm_date']]
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


def pca(data):
    model = PCA(n_components=2, random_state=0)
    data2 = model.fit_transform(data)
    return data2


def viz_samples(data, response, s):
    df = pd.DataFrame(data)
    df.columns = ['x', 'y']
    df['z'] = response
    sns.lmplot('x', 'y', data=df, fit_reg=False, hue="z", scatter_kws={"marker": "o", "s": 10})
    plt.title('Scatter Plot of Patient Populations:' + s)
    plt.xlabel('x')
    plt.ylabel('y')
    return df


def make_predictions(train_x, train_y, test_x, param):
    if param[2] == 'rf':
        clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=param[1], random_state=0, class_weight='balanced')
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


def get_transition_counts(dm, control, vars):
    dm = create_subwindows(dm, 0)
    control = create_subwindows(control, 0)
    df = pd.concat([dm, control], axis=0)
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
    with open('./data/data_all4comorbidity.pickle', 'wb') as f:
        pickle.dump(data, f)
    f.close()
    # find patients with diabetes: dxcat = '49' or '50'
    data_dm, ptids_dm = find_visit_gaps(data, ['49', '50'], 'dm')
    ptids_dm2 = find_patient_counts(data_dm)
    data_dm2 = data_dm[data_dm['ptid'].isin(ptids_dm2)]
    # get the visits in the observation window of the target patients
    data_dm3 = data_dm2[data_dm2['gap_dm'].between(180 * 24 * 60, 540 * 24 * 60)]
    ptids_dm3 = set(data_dm3['ptid']) # 4534 pts

    # find patients with CHF: dxcat = '108'
    data_chf, ptids_chf = find_visit_gaps(data, ['108'], 'chf')
    # ptids_chf2 = find_patient_counts(data_chf)

    # find patients with CKD: dxcat = '158'
    data_ckd, ptids_ckd = find_visit_gaps(data, ['158'], 'ckd')
    # find_patient_counts(data_ckd)

    # find patients with CKD: dxcat = '127'
    data_copd, ptids_copd = find_visit_gaps(data, ['127'], 'copd')
    # find_patient_counts(data_copd)

    # find patients with Comorbidities: dxcat = ['49', '50', '108', '158', '127']
    data_comorb, ptids_comorb = find_visit_gaps(data, ['49', '50', '108', '158', '127'], 'comorb')

    with open('./data/data_chf_ptids.pickle', 'wb') as f:
        pickle.dump([data_chf, ptids_chf], f)
    f.close()
    with open('./data/data_dm_ptids.pickle', 'wb') as f:
        pickle.dump([data_dm, ptids_dm], f)
    f.close()
    with open('./data/data_ckd_ptids.pickle', 'wb') as f:
        pickle.dump([data_ckd, ptids_ckd], f)
    f.close()
    with open('./data/data_copd_ptids.pickle', 'wb') as f:
        pickle.dump([data_copd, ptids_chf], f)
    f.close()
    with open('./data/data_comorb_ptids.pickle', 'wb') as f:
        pickle.dump([data_comorb, ptids_comorb], f)
    f.close()

    # find patients with at least four years of complete visits
    # 1. first visit date = 0
    # 2. one year of observation window and three years of prediction window
    thres = 60 * 24 * 360 * 4
    data_control = find_visit_gaps_control(data, ptids_dm, thres)
    data_control2 = data_control[data_control['adm_date'] <= 24 * 60 * 360]
    data_control3 = data_control2[data_control2['dis_date'] <= 24 * 60 * 360]
    ptids_control = set(data_control3['ptid']) # 30751 pts

    # get the counts of dxcats of patients
    counts_dm = get_counts_by_class(data_dm3, 1, len(ptids_dm3) * 0.05)
    counts_control = get_counts_by_class(data_control3, 0, len(ptids_control) * 0.05)
    counts = counts_dm.append(counts_control).fillna(0)
    prelim_features = set(counts.columns[:-1]) #34

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
    clf0, features_wts0, results_by_f0, results_by_auc0 = make_prediction_and_tuning(train_x0, train_y0, test_x0, test_y0, features0, [1000, 15, 2, 'rf'])
    clf0, features_wts0, results_by_f0, results_by_auc0 = make_prediction_and_tuning(train_x0, train_y0, test_x0, test_y0, features0, [0.01, 15, 2, 'lr'])
    # auc: 0.575, 0.629
    # ============= baseline 2: frequency in sub-window ===================================
    # every season: get the counts and then append
    counts_sub_dm = get_counts_subwindow(data_dm3, 1, prelim_features, 3)
    counts_sub_control = get_counts_subwindow(data_control3, 0, prelim_features, 3)
    counts_sub = counts_sub_dm.append(counts_sub_control).fillna(0)
    counts_sub.to_csv('./data/counts_sub_by3month.csv')

    counts_sub_dm = get_counts_subwindow(data_dm3, 1, prelim_features, 2)
    counts_sub_control = get_counts_subwindow(data_control3, 0, prelim_features, 2)
    counts_sub = counts_sub_dm.append(counts_sub_control).fillna(0)
    counts_sub.to_csv('./data/counts_sub_by2month.csv')

    counts_sub_dm = get_counts_subwindow(data_dm3, 1, prelim_features, 1)
    counts_sub_control = get_counts_subwindow(data_control3, 0, prelim_features, 1)
    counts_sub = counts_sub_dm.append(counts_sub_control).fillna(0)
    counts_sub.to_csv('./data/counts_sub_by1month.csv')

    counts_sub_dm = get_counts_subwindow(data_dm3, 1, prelim_features, 6)
    counts_sub_control = get_counts_subwindow(data_control3, 0, prelim_features, 6)
    counts_sub = counts_sub_dm.append(counts_sub_control).fillna(0)
    counts_sub.to_csv('./data/counts_sub_by6month.csv')

    counts_sub_x = counts_sub[counts_sub.columns[:-1]]
    counts_sub_y = counts_sub['response']
    features1 = counts_sub.columns.tolist()[:-1]
    train_x1, train_y1, test_x1, test_y1 = split_train_test_sets(train_ids, test_ids, counts_sub_x, counts_sub_y)
    clf1, features_wts1, results_by_f1, results_by_auc1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features1, [1000, 15, 2, 'rf'])
    clf1, features_wts1, results_by_f1, results_by_auc1 = make_prediction_and_tuning(train_x1, train_y1, test_x1, test_y1, features1, [0.01, 15, 2, 'lr'])

    # ============== baseline 3: mining sequence patterns =============================================
    # get the sequence by sub-windows
    seq_dm = create_sequence(data_dm4, 'dm_train')
    seq_control = create_sequence(data_control4, 'control_train')
    # load the selected features using sequential pattern mining SPADE
    features2_dm = pd.read_csv('./data/result_dm.txt', delimiter=' -1', header=None)
    # features2_control = pd.read_csv('./data/result_control.txt', delimiter=' -1', header=None)
    # merge the features selected to get the selected single features
    features2a = features2_dm[0].values.tolist()[:-2] + ['167', '663']
    features2a_names = ['cat' + i for i in features2a] + ['response']
    counts_bps = counts[features2a_names]

    cooccur_list = [[258, 259], [53, 98], [204, 211]]
    mvisit_list = [259, 211]
    counts_bpsb = get_seq_item_counts(seq_dm, seq_control, cooccur_list, mvisit_list)
    counts_bps = pd.concat([counts_bpsb, counts], axis=1).fillna(0)
    counts_bps.to_csv('./data/counts_bps.csv')

    counts_bps_y = counts_bps['response']
    counts_bps_x = counts_bps
    del counts_bps_x['response']
    features2 = counts_bps_x.columns.tolist()
    train_x2, train_y2, test_x2, test_y2 = split_train_test_sets(train_ids, test_ids, counts_bps_x, counts_bps_y)
    clf2, features_wts2, results_by_f2, results_by_auc2 = make_prediction_and_tuning(train_x2, train_y2, test_x2, test_y2, features2, [1000, 15, 2, 'rf'])
    clf2, features_wts2, results_by_f2, results_by_auc2 = make_prediction_and_tuning(train_x2, train_y2, test_x2, test_y2, features2, [0.01, 15, 2, 'lr'])

    # ============== another baseline 4: transitions ====================================================
    counts_trans = get_transition_counts(data_dm4, data_control4, prelim_features)
    counts_trans = pd.concat([counts_trans, counts], axis=1).fillna(0)
    counts_trans.to_csv('./data/counts_trans.csv')

    counts_trans_x = counts_trans[counts_trans.columns[:-1]]
    counts_trans_y = counts_trans['response']
    features4 = counts_trans.columns.tolist()[:-1]
    train_x4, train_y4, test_x4, test_y4 = split_train_test_sets(train_ids, test_ids, counts_trans_x, counts_trans_y)
    clf4, features_wts4, results_by_f4, results_by_auc4 = make_prediction_and_tuning(train_x4, train_y4, test_x4, test_y4, features4, [1000, 15, 2, 'rf'])
    clf4, features_wts4, results_by_f4, results_by_auc4 = make_prediction_and_tuning(train_x4, train_y4, test_x4, test_y4, features4, [0.01, 15, 2, 'lr'])

    # ============= Proposed: frequency in sub-window and selected by sgl===================================
    # train_ids = pd.read_csv('./data/train_ids.csv', header=None, dtype=object)
    # train_ids = train_ids.values.flatten()
    # test_ids = pd.read_csv('./data/test_ids.csv', header=None, dtype=object)
    # test_ids = test_ids.values.flatten()

    counts_sub = pd.read_csv('./data/counts_sub_by3month.csv')
    counts_sub.index = counts_sub['ptid'].astype(str)
    del counts_sub['ptid']
    data_cols = counts_sub.columns
    alphas = np.arange(0, 1.1, 0.1)
    for a in alphas:
        print('When alpha = %.1f:' % a)
        features3_all = pd.read_csv('./data/sgl_coefs_4group_alpha' + str(int(a * 10)) + '_v2.csv')
        del features3_all['Unnamed: 0']
        feature_names_sgl = []
        for i in features3_all.columns:
            # print('The %i-th lambda:' % i)
            features3_inds = features3_all[i]
            features3 = [(data_cols[j], features3_inds.loc[j]) for j in features3_inds.index if features3_inds.loc[j] != 0]
            features3 = sorted(features3, key=itemgetter(1), reverse=True)
            feature_names_sgl.append(features3)
            features3_pos = [(data_cols[j], features3_inds.loc[j]) for j in features3_inds.index if features3_inds.loc[j] > 0]
            if len(features3) > 0:
                print('%i features are selected in SGL, %i are positive' % (len(features3), len(features3_pos)))
                # counts_sgl_x = counts_sub[features3]
                # counts_sgl_y = counts_sub['response']
                # train_x3, train_y3, test_x3, test_y3 = split_train_test_sets(train_ids, test_ids, counts_sgl_x, counts_sgl_y)
                # clf3, features_wts3, results_by_f3, results_by_auc3 = make_prediction_and_tuning(train_x3, train_y3, test_x3, test_y3, features3, [1000, 15, 2, 'rf'])
                # clf3, features_wts3, results_by_f3, results_by_auc3 = make_prediction_and_tuning(train_x3, train_y3, test_x3, test_y3, features3, [0.01, 15, 2, 'lr'])
            else:
                print('No feature is selected in SGL!')
    a = 0.8
    features3_all = pd.read_csv('./data/sgl_coefs_4group_alpha' + str(int(a * 10)) + '_v2.csv')
    del features3_all['Unnamed: 0']
    i = features3_all.columns[5]
    features3_inds = features3_all[i]
    features3 = [data_cols[j] for j in features3_inds.index if features3_inds.loc[j] != 0]
    counts_sgl_x = counts_sub[features3]
    counts_sgl_y = counts_sub['response']
    train_x3, train_y3, test_x3, test_y3 = split_train_test_sets(train_ids, test_ids, counts_sgl_x, counts_sgl_y)
    clf3, features_wts3, results_by_f3, results_by_auc3 = make_prediction_and_tuning(train_x3, train_y3, test_x3, test_y3, features3, [1000, 15, 2, 'rf'])
    clf3, features_wts3, results_by_f3, results_by_auc3 = make_prediction_and_tuning(train_x3, train_y3, test_x3, test_y3, features3, [0.01, 15, 2, 'lr'])

    features3 = [(data_cols[j], features3_inds.loc[j]) for j in features3_inds.index if features3_inds.loc[j] != 0]
    features3 = sorted(features3, key=itemgetter(1), reverse=True)

    with open('./data/train_test_ptids.pickle', 'rb') as f:
        train_ids, test_ids = pickle.load(f)
    f.close()

    # ======================================= collect prediction results =============================

    counts = pd.read_csv('./data/dm_control_counts.csv')
    counts.index = counts['ptid'].astype(str)
    del counts['ptid']
    counts_x = counts[counts.columns[:-1]]
    counts_y = counts['response']
    features0 = counts.columns.tolist()[:-1]
    train_x0, train_y0, test_x0, test_y0 = split_train_test_sets(train_ids, test_ids, counts_x, counts_y)

    counts_sub = pd.read_csv('./data/counts_sub_by3month.csv')
    counts_sub.index = counts_sub['ptid'].astype(str)
    del counts_sub['ptid']
    counts_sub_x = counts_sub[counts_sub.columns[:-1]]
    counts_sub_y = counts_sub['response']
    features1 = counts_sub.columns.tolist()[:-1]
    train_x1, train_y1, test_x1, test_y1 = split_train_test_sets(train_ids, test_ids, counts_sub_x, counts_sub_y)

    counts_bps = pd.read_csv('./data/counts_bps.csv')
    counts_bps.columns = ['ptid'] + list(counts_bps.columns[1:])
    counts_bps.index = counts_bps['ptid'].astype(str)
    del counts_bps['ptid']
    counts_bps_x = counts_bps[counts_bps.columns[:-1]]
    counts_bps_y = counts_bps['response']
    features2 = counts_bps.columns.tolist()[:-1]
    train_x2, train_y2, test_x2, test_y2 = split_train_test_sets(train_ids, test_ids, counts_bps_x, counts_bps_y)

    counts_trans = pd.read_csv('./data/counts_trans.csv')
    counts_trans.columns = ['ptid'] + list(counts_trans.columns[1:])
    counts_trans.index = counts_trans['ptid'].astype(str)
    del counts_trans['ptid']
    counts_trans_x = counts_trans[counts_trans.columns[:-1]]
    counts_trans_y = counts_trans['response']
    features4 = counts_trans.columns.tolist()[:-1]
    train_x4, train_y4, test_x4, test_y4 = split_train_test_sets(train_ids, test_ids, counts_trans_x, counts_trans_y)

    a = 0.8
    features3_all = pd.read_csv('./data/sgl_coefs_4group_alpha' + str(int(a * 10)) + '_v2.csv')
    del features3_all['Unnamed: 0']
    i = features3_all.columns[5]
    features3_inds = features3_all[i]
    features3 = [data_cols[j] for j in features3_inds.index if features3_inds.loc[j] != 0]
    counts_sgl_x = counts_sub[features3]
    counts_sgl_y = counts_sub['response']
    train_x3, train_y3, test_x3, test_y3 = split_train_test_sets(train_ids, test_ids, counts_sgl_x, counts_sgl_y)


    # get the trans
    test_proba0a = make_predictions(train_x0, train_y0, test_x0, [1000, 15, 'rf'])
    test_proba0b = make_predictions(train_x0, train_y0, test_x0, [1000, 15, 'lr'])
    test_proba0c = make_predictions(train_x0, train_y0, test_x0, [0.01, 15, 'lr'])

    test_proba1a = make_predictions(train_x1, train_y1, test_x1, [1000, 15, 'rf'])
    test_proba1b = make_predictions(train_x1, train_y1, test_x1, [1000, 15, 'lr'])
    test_proba1c = make_predictions(train_x1, train_y1, test_x1, [0.01, 15, 'lr'])

    test_proba2a = make_predictions(train_x2, train_y2, test_x2, [1000, 15, 'rf'])
    test_proba2b = make_predictions(train_x2, train_y2, test_x2, [1000, 15, 'lr'])
    test_proba2c = make_predictions(train_x2, train_y2, test_x2, [0.01, 15, 'lr'])

    test_proba3a = make_predictions(train_x4, train_y4, test_x4, [1000, 15, 'rf'])
    test_proba3b = make_predictions(train_x4, train_y4, test_x4, [1000, 15, 'lr'])
    test_proba3c = make_predictions(train_x4, train_y4, test_x4, [0.01, 15, 'lr'])

    test_proba4a = make_predictions(train_x3, train_y3, test_x3, [1000, 15, 'rf'])
    test_proba4b = make_predictions(train_x3, train_y3, test_x3, [1000, 15, 'lr'])
    test_proba4c = make_predictions(train_x3, train_y3, test_x3, [0.01, 15, 'lr'])

    test_proba = pd.DataFrame([test_proba0a, test_proba0b, test_proba0c, test_y0.values.tolist(),
                               test_proba1a, test_proba1b, test_proba1c, test_y1.values.tolist(),
                               test_proba2a, test_proba2b, test_proba2c, test_y2.values.tolist(),
                               test_proba3a, test_proba3b, test_proba3c, test_y4.values.tolist(),
                               test_proba4a, test_proba4b, test_proba4c, test_y3.values.tolist()])
    test_proba = test_proba.transpose()
    test_proba.columns = ['b1_rf', 'b1_lr', 'b1_lasso', 'b1_response', 'b2_rf', 'b2_lr', 'b2_lasso', 'b2_response',
                          'b3_rf', 'b3_lr', 'b3_lasso', 'b3_response', 'b4_rf', 'b4_lr', 'b4_lasso', 'b4_response',
                          'b5_rf', 'b5_lr', 'b5_lasso', 'b5_response']
    test_proba.to_csv('./data/test_proba_v3.csv', index=False)

    # calculate F1 score
    test_proba = pd.read_csv('./data/test_proba_v3.csv')
    cols = [['b1_rf', 'b1_lasso', 'b1_response', 0.47, 0.11],
            ['b3_rf', 'b3_lasso', 'b3_response', 0.47, 0.11],
            ['b4_rf', 'b4_lasso', 'b4_response', 0.51, 0.11],
            ['b5_rf', 'b5_lasso', 'b5_response', 0.41, 0.15]]
    # f1s = []
    # for i in cols:
    #     res_rf1 = [1 if p >= i[3] else 0 for p in test_proba[i[0]].values]
    #     res_rf0 = [1 if p < i[3] else 0 for p in test_proba[i[0]].values]
    #     f1_rf1 = metrics.fbeta_score(test_proba[i[2]].values, res_rf1, beta=3)
    #     f1_rf0 = metrics.fbeta_score(test_proba[i[2]].values, res_rf0, beta=3)
    #     res_lr = [1 if p >= i[4] else 0 for p in test_proba[i[1]].values]
    #     f1_lr = metrics.fbeta_score(test_proba[i[2]].values, res_lr, beta=3)
    #     f1s.append(((f1_lr1+f1_lr0)/2, f1_lr))

    def calculate_fscores_bootstraps(test_proba, thres, y):
        res = [1 if p >= thres else 0 for p in test_proba]
        f2s = []
        for p in range(50):
            random.seed(p)
            sp = random.choices(list(zip(res, y)), k=int(len(y)))
            sp_pred = [v[0] for v in sp]
            sp_true = [v[1] for v in sp]
            f2 = metrics.fbeta_score(sp_true, sp_pred, average='binary', beta=2)
            f2s.append(f2)
        avg = np.mean(f2s)
        std = np.std(f2s)
        return tuple((avg, std))

    fs = []
    for i in cols:
        results_rf = calculate_fscores_bootstraps(test_proba[i[0]].values, i[3], test_proba[i[2]].values)
        results_lr = calculate_fscores_bootstraps(test_proba[i[1]].values, i[4], test_proba[i[2]].values)
        fs.append([results_lr, results_rf])



    data_dm4[data_dm4['ptid'] == '769052'].to_csv('./data/example_dmpt.csv') # rf predicted proba: 0.782
    data_control4[data_control4['ptid'] =='1819093'].to_csv('./data/example_controlpt.csv') # rf predicted proba: 0.033

    data['adm_day'] = data['adm_date'].apply(lambda x: int(x / 60 / 24))
    data['adm_month'] = data['adm_day'].apply(lambda x: int(x / 30))
    del data['rank']
    del data['dis_date']
    del data['adm_date']
    # analysis on example patient
    # import pandas as pd
    # import pickle
    # test_proba = pd.read_csv('./data/test_proba_v3.csv')
    #
    # with open('./data/train_test_ptids.pickle', 'rb') as f:
    #     train_ids, test_ids = pickle.load(f)
    #
    # test_proba['ptid'] = test_ids
    # test_proba.sort(['b5_rf', 'b5_lasso'], ascending=[1, 1], inplace=True)
    #
    # with open('./data/visits_v4.pickle', 'rb') as f:
    #     visits = pickle.load(f)
    #
    # visits.columns = ['ptid', 'vid', 'IPorOP', 'adm_date', 'dis_date', 'rank']
    # with open('./data/dxs_data_v2.pickle', 'rb') as f:
    #     data_dx = pickle.load(f)
    #
    #
    # dxgrps, dxgrps_dict, dxgrps_dict2 = dx2dxcat()
    # data_dx2 = process_dxs(data_dx, dxgrps_dict, dxgrps_dict2)
    # data_dx2.head()
    #
    # data = merge_visit_and_dx(data_dx2,
    #                           visits)
    #
    # # exm1 = data[data['ptid']=='1094444']
    # # exm1 = data[data['ptid']=='750517']
    # exm1 = data[data['ptid']=='1407189']
    # exm1['adm_day'] = exm1['adm_date'].apply(lambda x: int(x / 60 / 24))
    # exm1['adm_month'] = exm1['adm_day'].apply(lambda x: int(x / 30))
    # del exm1['rank']
    # del exm1['dis_date']
    # del exm1['adm_date']


# ============= Add t-sne or pca for visualization ==========================================================

    train_tsne0 = tsne(train_x0)
    test_tsne0 = tsne(test_x0)
    train_tsne1 = tsne(train_x1)
    test_tsne1 = tsne(test_x1)
    train_tsne2 = tsne(train_x2)
    test_tsne2 = tsne(test_x2)

    output_train_tsne0 = viz_samples(train_tsne0, train_y0.values, 'baseline-0-train')
    output_test_tsne0 = viz_samples(test_tsne0, test_y0.values, 'baseline-0-test')

    output_train_tsne1 = viz_samples(train_tsne1, train_y1.values, 'baseline-1-train')
    output_test_tsne1 = viz_samples(test_tsne1, test_y1.values, 'baseline-1-test')

    output_train_tsne2 = viz_samples(train_tsne2, train_y2.values, 'baseline-2-train')
    output_test_tsne2 = viz_samples(test_tsne2, test_y2.values, 'baseline-2-test')

    # output_train_tsne3 = viz_tsne(train_x3, train_y3, 'baseline-3-train')
    # output_test_tsne3 = viz_tsne(test_x3, test_y3, 'baseline-3-test')

    train_pca0 = pca(train_x0)
    test_pca0 = pca(test_x0)
    train_pca1 = pca(train_x1)
    test_pca1 = pca(test_x1)
    train_pca2 = pca(train_x2)
    test_pca2 = pca(test_x2)

    output_train_tsne1 = viz_samples(train_pca1, train_y1.values, 'baseline-1-train')
    output_test_tsne1 = viz_samples(test_pca1, test_y1.values, 'baseline-1-test')

    output_train_pca0 = viz_samples(train_pca0, train_y0.values, 'baseline-0-train')
    output_test_pca0 = viz_samples(test_pca0, test_y0.values, 'baseline-0-test')

    output_train_pca1 = viz_samples(train_pca1, train_y1.values, 'baseline-1-train')
    output_test_pca1 = viz_samples(test_pca1, test_y1.values, 'baseline-1-test')

    output_train_pca2 = viz_samples(train_pca2, train_y2.values, 'baseline-2-train')
    output_test_pca2 = viz_samples(test_pca2, test_y2.values, 'baseline-2-test')