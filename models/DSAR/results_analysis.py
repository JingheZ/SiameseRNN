# analyze the results
import random
from sklearn import metrics
import numpy as np
import pickle


def calculate_scores_bootstraps(pred, y, val):
    f2s = []
    recalls = []
    specificity = []
    aucs = []
    for p in range(50):
        random.seed(p)
        sp = random.choices(list(zip(pred, y, val)), k=int(len(y)))
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
    avg1 = np.mean(f2s)
    std1 = np.std(f2s)
    avg2 = np.mean(recalls)
    std2 = np.std(recalls)
    avg3 = np.mean(specificity)
    std3 = np.std(specificity)
    avg4 = np.mean(aucs)
    std4 = np.std(aucs)
    return [(avg2, std2), (avg3, std3), (avg4, std4), (avg1, std1)]


def calculate_results(result_file):
    with open(result_file, 'rb') as f:
        pred, val, y = pickle.load(f)
    res = calculate_scores_bootstraps(pred, y, val)
    return res


# =============================== LR ================================
model_type = 'LR'
result_file = './results/test_results_' + model_type + '.pickle'
lr = calculate_results(result_file)

# =============================== MLP ===============================
model_type = 'MLP-256'
result_file = './results/test_results_' + model_type + '.pickle'
mlp = calculate_results(result_file)

# =============================== p2v ===============================
model_type = 'crnn2-bi-tanh-fn'
result_file = './results/test_results_' + model_type + '_layer1.pickle'
p2v = calculate_results(result_file)

# =============================== rnn mge ===========================
model_type = 'rnn'
result_file = './results/test_results_' + model_type + '_layer1.pickle'
rnn_mge = calculate_results(result_file)

# =============================== bi-rnn mge ===========================
model_type = 'rnn-bi'
result_file = './results/test_results_' + model_type + '_layer1.pickle'
birnn_mge = calculate_results(result_file)

# =============================== retain ===========================
model_type = 'retain'
result_file = './results/test_results_' + model_type + '_layer1.dat'
retain = calculate_results(result_file)


# =============================== rnn mve ===========================
model_type = 'rnn'
result_file = './results/test_results_w2v_' + model_type + '_layer1.pickle'
rnn_mve = calculate_results(result_file)

# =============================== bi-rnn mve ===========================
model_type = 'rnn-bi'
result_file = './results/test_results_w2v_' + model_type + '_layer1.pickle'
birnn_mve = calculate_results(result_file)