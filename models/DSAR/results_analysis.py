# analyze the results
import random
from sklearn import metrics
import numpy as np
import pickle
import pandas as pd


def calculate_scores_bootstraps(pred, y, val):
    f2s = []
    recalls = []
    specificity = []
    aucs = []
    f1s = []
    for p in range(50):
        # random.seed(p)
        # sp = random.choices(list(zip(pred, y, val)), k=int(len(y)))
        sp = list(zip(pred, y, val))
        sp_pred = [v[0] for v in sp]
        sp_true = [v[1] for v in sp]
        sp_val = [v[2] for v in sp]
        f2 = metrics.fbeta_score(sp_true, sp_pred, average='binary', beta=2)
        f2s.append(f2)
        rec = metrics.recall_score(sp_true, sp_pred, pos_label=1)
        recalls.append(rec)
        spec = metrics.recall_score(sp_true, sp_pred, pos_label=0)
        specificity.append(spec)
        auc = metrics.roc_auc_score(sp_true, sp_val)
        aucs.append(auc)
        f1 = metrics.fbeta_score(sp_true, sp_pred, average='binary', beta=1)
        f1s.append(f1)
    avg1 = np.mean(f2s)
    std1 = np.std(f2s)
    avg2 = np.mean(recalls)
    std2 = np.std(recalls)
    avg3 = np.mean(specificity)
    std3 = np.std(specificity)
    avg4 = np.mean(aucs)
    std4 = np.std(aucs)
    avg5 = np.mean(f1s)
    std5 = np.std(f1s)
    return [(avg2, std2), (avg3, std3), (avg4, std4), (avg1, std1), (avg5, std5)]


def calculate_results(result_file):
    with open(result_file, 'rb') as f:
        pred, val, y = pickle.load(f)
    res = calculate_scores_bootstraps(pred, y, val)
    return res


def analyze_example_pts(result_file):
    with open(result_file, 'rb') as f:
        pred, val, y = pickle.load(f)
    with open('./data/hospitalization_train_validate_test_ids.pickle', 'rb') as f:
        train_ids, valid_ids, test_ids = pickle.load(f)
    result = list(zip(test_ids, pred, y, val))
    result = pd.DataFrame(result)
    result.columns = ['ptid', 'pred_label', 'true_label', 'pred_val']
    result = result.sort_values(['true_label', 'pred_val'], ascending=[0, 0])

    l = 3
    with open('./data/clinical_events_hospitalization.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('./data/hospitalization_test_data_by_' + str(l) + 'month.pickle', 'rb') as f:
        test, test_y = pickle.load(f)
    f.close()
    batch_x, batch_demoip, _ = create_batch(i, batch_size, test, test_demoips, test_y, w2v, vsize, pad_size, l)
    y_pred, _, _ = model(batch_x, batch_demoip, batch_size)



# =============================== LR ================================
model_type = 'LR'
result_file = './results/test_results_' + model_type + '.pickle'
lr = calculate_results(result_file)

# =============================== MLP ===============================
model_type = 'MLP-256'
result_file = './results/test_results_' + model_type + '.pickle'
mlp = calculate_results(result_file)

# =============================== rnn mge ===========================
model_type = 'rnn'
result_file = './results/test_results_cts_' + model_type + '_layer1.pickle'
rnn_mge = calculate_results(result_file)

# =============================== bi-rnn mge ===========================
model_type = 'rnn-bi'
result_file = './results/test_results_cts_' + model_type + '_layer1.pickle'
birnn_mge = calculate_results(result_file)

# =============================== retain ===========================
model_type = 'retain'
result_file = './results/test_results_' + model_type + '_layer1.pickle'
retain = calculate_results(result_file)

# =============================== rnn mve ===========================
model_type = 'rnn'
result_file = './results/test_results_w2v_' + model_type + '_layer1.pickle'
rnn_mve = calculate_results(result_file)

# =============================== bi-rnn mve ===========================
model_type = 'rnn-bi'
result_file = './results/test_results_w2v_' + model_type + '_layer1.pickle'
birnn_mve = calculate_results(result_file)

# =============================== p2v ===============================
model_type = 'crnn2-bi-tanh-fn'
result_file = './results/test_results_' + model_type + '_layer1.pickle'
p2v = calculate_results(result_file)
result_file = './results/test_results_w2v_' + model_type + '_layer1_nf10_a01_v2.pickle'
p2v = calculate_results(result_file)
