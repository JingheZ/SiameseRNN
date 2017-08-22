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


def tune_proba_threshold(pred_proba, y, b):
    pred_proba = [i[1] for i in pred_proba]
    results = []
    for t in np.arange(0, 1, 0.01):
        res = [1 if p > t else 0 for p in pred_proba]
        f1 = metrics.fbeta_score(y, res, beta=b)
        results.append((t, f1))
        # auc = metrics.roc_auc_score(y, res)
        # results.append((t, auc))
    threshold = max(results, key=itemgetter(1))[0]
    return threshold, results


def make_prediction_and_tuning(train_x, train_y, test_x, test_y, param):
    clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', n_jobs=10, random_state=0)
    # clf = None
    # if s == 'svm':
    #     clf = SVC(kernel=param[0], class_weight='balanced', probability=True)
    #     # clf = SVC(kernel=param[0])
    # elif s == 'rf':
    #     clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy', class_weight='balanced')
    #     # clf = RandomForestClassifier(n_estimators=param[0], criterion='entropy')
    # elif s == 'lda':
    #     clf = LinearDiscriminantAnalysis()
    # elif s == 'knn':
    #     clf = neighbors.KNeighborsClassifier(param[0], weights='distance')
    # # elif s == 'xgb':
    # #     param_dist = {'objective': 'binary:logistic', 'n_estimators': param[0], 'learning_rate': param[1]}
    # #     xgb.XGBClassifier(**param_dist)
    clf.fit(train_x, train_y)
    pred_train = clf.predict_proba(train_x)
    pred_test = clf.predict_proba(test_x)
    pred_proba = [i[1] for i in pred_test]
    threshold, tuning = tune_proba_threshold(pred_train, train_y, 3) # 2.5
    pred = [1 if p > threshold else 0 for p in pred_proba]
    result = metrics.classification_report(test_y, pred)
    auc = metrics.roc_auc_score(test_y, pred)
    print(auc)
    print(result)
    return pred, result, auc


def split_train_test(X, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    X = X.values
    y = y.values
    for train_index, test_index in sss.split(X, y):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
    return train_x, train_y, test_x, test_y


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
    data_dm3 = data_dm2[data_dm2['gap_dm'].between(90 * 24 * 60, 455 * 24 * 60)]
    ptids_dm3 = set(data_dm3['ptid']) # 5664 pts
    # # find patients with CHF: dxcat = '108'
    # data_chf = find_visit_gaps(data, ['108'])
    # find_patient_counts(data_chf)
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
    # 2. one year of observation window and three years of prediction window
    thres = 60 * 24 * 365 * 3
    data_control = find_visit_gaps_control(data, ptids_dm, thres)
    data_control2 = data_control[data_control['adm_date'] <= 24 * 60 * 365]
    data_control3 = data_control2[data_control2['dis_date'] <= 24 * 60 * 365]
    ptids_control = set(data_control3['ptid']) # 47899 pts

    # get the counts of dxcats of patients
    counts_dm = get_counts_by_class(data_dm3, 1, 57)
    counts_control = get_counts_by_class(data_control3, 0, 479)
    counts = counts_dm.append(counts_control).fillna(0)
    counts.columns = ['cat' + i for i in counts.columns[:-1]] + ['response']
    counts.to_csv('./data/dm_control_counts.csv')

    # get counts and do preliminary feature selection
    counts_x, counts_y, features = feature_selection_prelim(counts, 50)
    # use actual ratio in training and testing:
    train_x, train_y, test_x, test_y = split_train_test(counts_x, counts_y)
    #
    # # use balanced data in training but actual ratio in testing
    # train_ids_pos, test_ids_pos = create_train_validate_test_sets_positive(np.array(list(ptids_dm3)))
    # test_ratio = len(ptids_control) / len(ptids_dm3)
    # train_ids_neg, test_ids_neg = create_train_validate_test_sets_negative(list(ptids_control), len(ptids_dm3), test_ratio, train_ratio=1)
    # train_ids = train_ids_pos + train_ids_neg
    # test_ids = test_ids_pos + test_ids_neg
    # train_x, train_y, test_x, test_y = create_experiment_data(counts_x, counts_y, train_ids, test_ids)

    # build training model and make predictions
    experiments(train_x, train_y, test_x, test_y)

