# analyze the results
import random
from sklearn import metrics
import numpy as np
import pickle
import pandas as pd
import operator
from sklearn.preprocessing import normalize
from collections import Counter


def calculate_scores_bootstraps(pred, y, val):
    f2s = []
    recalls = []
    specificity = []
    aucs = []
    f1s = []
    for p in range(20):
        # random.seed(p)
        sp = random.choices(list(zip(pred, y, val)), k=int(len(y)))
        # sp = list(zip(pred, y, val))
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


def get_csv(result_file):
    with open(result_file, 'rb') as f:
        pred, val, y = pickle.load(f)
    sp = list(zip(pred, y, val))
    result = pd.DataFrame(sp)
    result.columns = ['pred_label', 'true_label', 'pred_val']
    result.to_csv('./data/result_' + model_type + '.csv', index=False)


def get_codes(test, ind, itemdict):
    data = test[ind]
    meds_seq = []
    for i in range(4):
        meds = []
        if len(data[i]) > 0:
            for j in data[i]:
                if itemdict.__contains__(j):
                    grp = itemdict[j]
                else:
                    grp = j
                meds.append(grp)
        meds_seq.append(meds)
    return meds_seq


def aggregate_code_wts(meds_seq, data):
    wts = []
    for i in range(4):
        w = {}
        if len(meds_seq[i]) > 0:
            for k in range(len(meds_seq[i])):
                if not w.__contains__(meds_seq[i][k]):
                    w[meds_seq[i][k]] = 0
                w[meds_seq[i][k]] += data[i][k]
            if w.__contains__('p0'):
                del w['p0']
            if w.__contains__('dx259'):
                del w['dx259']
            wk = list(w.keys())
            wv = list(w.values())
            wv = normalize(wv, norm='l1').tolist()[0]
            w1 = dict(zip(wk, wv))
            w = sorted(w1.items(), key=operator.itemgetter(1), reverse=True)
        wts.append(w)
    return wts


def aggregate_code_wts_items_top(meds_seq, seq_wts, data, n):
    w = {}
    for i in range(4):
        if len(meds_seq[i]) > 0:
            for k in range(len(meds_seq[i])):
                if not w.__contains__(meds_seq[i][k]):
                    w[meds_seq[i][k]] = 0
                w[meds_seq[i][k]] += data[i][k] * seq_wts[i]
    if w.__contains__('p0'):
        del w['p0']
    if w.__contains__('dx259'):
        del w['dx259']
    wk = list(w.keys())
    wv = list(w.values())
    wv = normalize(wv, norm='l1').tolist()[0]
    w1 = dict(zip(wk, wv))
    wts = sorted(w1.items(), key=operator.itemgetter(1), reverse=True)[:n]
    return wts


if __name__ == '__main__':
    # ======================== Prediction performance ==========================================
    # ===== LR =====
    model_type = 'LR'
    result_file = './results/test_results_' + model_type + '.pickle'
    lr = calculate_results(result_file)
    get_csv(result_file)
    # ===== MLP =====
    model_type = 'MLP-256'
    result_file = './results/test_results_' + model_type + '.pickle'
    mlp = calculate_results(result_file)

    # ===== rnn mge ==
    model_type = 'rnn'
    # result_file = './results/test_results_cts_' + model_type + '_layer1.pickle'
    result_file = './results/test_results_cts_' + model_type + '_layer1V2.pickle'
    rnn_mge = calculate_results(result_file)

    # ==== bi-rnn mge ====
    model_type = 'rnn-bi'
    result_file = './results/test_results_cts_' + model_type + '_layer1V2.pickle'
    birnn_mge = calculate_results(result_file)

    # ===== retain =======
    model_type = 'retain'
    result_file = './results/test_results_' + model_type + '_layer1.pickle'
    retain = calculate_results(result_file)

    # === rnn mve =======
    model_type = 'rnn'
    result_file = './results/test_results_w2v_' + model_type + '_layer1.pickle'
    rnn_mve = calculate_results(result_file)

    # ======= bi-rnn mve ====
    model_type = 'rnn-bi'
    result_file = './results/test_results_w2v_' + model_type + '_layer1.pickle'
    birnn_mve = calculate_results(result_file)

    # ====== p2v =======
    model_type = 'crnn2-bi-tanh-fn'
    result_file = './results/test_results_w2v_' + model_type + '_layer1_1.pickle'
    p2v = calculate_results(result_file)
    # result_file = './results/test_results_w2v_' + model_type + '_layer1_nf10_a01_v2.pickle'
    # p2v = calculate_results(result_file)
    #
    # result_file = './results/test_results_w2v_crnn2-bi-tanh-fn_layer1a001_saved.pickle'
    # p2v = calculate_results(result_file)

    # ============================= Interpretation of example pts =======================
    with open(result_file, 'rb') as f:
        pred, val, y = pickle.load(f)
    with open('./data/hospitalization_train_validate_test_ids.pickle', 'rb') as f:
        train_ids, valid_ids, test_ids = pickle.load(f)
    with open('./data/hospitalization_test_data_demoip.pickle', 'rb') as f:
        test_genders, test_ages, test_ip = pickle.load(f)
    f.close()

    result = list(zip(test_ids, pred, y, val))
    result = pd.DataFrame(result)
    result.columns = ['ptid', 'pred_label', 'true_label', 'pred_val']
    result['gender'] = test_genders
    result['age'] = test_ages
    result['ip'] = test_ip

    # get number of subseq
    l = 3
    with open('./data/clinical_events_hospitalization.pickle', 'rb') as f:
        data = pickle.load(f)
    data = data[data['ptid'].isin(test_ids)]
    data = data.sort(['ptid', 'adm_month'], ascending=[1, 1])
    data2 = data[data['adm_month'].isin((0, 11))]
    data2['adm_subseq'] = data2['adm_month'].apply(lambda x: int(x / l))
    subseq_cts = data2[['ptid', 'adm_subseq']].drop_duplicates().groupby('ptid').count()
    subseq_cts.reset_index(inplace=True)

    result = pd.merge(result, subseq_cts, left_on='ptid', right_on='ptid', how='inner')
    result_neg = result[result['true_label'] == 0]
    result = result.sort(['adm_subseq'], ascending=[0])

    # get primary pdx for hospitalization
    with open('./data/dxs_data_v2.pickle', 'rb') as f:
        dxs = pickle.load(f)
    f.close()

    data_ip = data[data['adm_month'] > 17]
    data_ip = data_ip[data_ip['cdrIPorOP'] == 'IP']
    data_ip = data_ip.sort(['adm_month'], ascending=[1])
    data_ip1 = data_ip[['ptid', 'adm_month']].groupby('ptid').min()
    data_ip1.reset_index(inplace=True)
    data_ip1.columns = ['ptid', 'first']
    data_ip2 = pd.merge(data_ip, data_ip1, left_on='ptid', right_on='ptid', how='inner')
    data_ip3 = data_ip2[data_ip2['adm_month'] == data_ip2['first']]

    data_ip_pdx = pd.merge(data_ip3[['ptid', 'vid', 'first']], dxs[['vid', 'pdx']].drop_duplicates(), left_on='vid',
                           right_on='vid', how='inner')
    data_ip_pdx = data_ip_pdx[['ptid', 'first', 'pdx']].drop_duplicates()
    data_ip_pdx.columns = ['ptid', 'ip_month', 'pdx_ip']
    result2 = pd.merge(result, data_ip_pdx, left_on='ptid', right_on='ptid', how='inner')

    chf = result2[result2['pdx_ip'].between('420', '4299')]
    copd = result2[result2['pdx_ip'].between('490', '496')]

    result2.to_csv('./results/result_table_pt_info_pos.csv', index=True)
    result_neg.to_csv('./results/result_table_pt_info_neg.csv', index=True)

    with open('./data/hospitalization_test_data_by_' + str(l) + 'month.pickle', 'rb') as f:
        test, test_y = pickle.load(f)
    f.close()
    # get example patient
    ptids1 = ['1446009', '1150550', '554938', '1348961', '1539136', '1703959']
    ptids0 = ['1790610', '1634904', '1218594', '773135', '1334050', '1596720']
    ptids = ptids1 + ptids0
    inds = [list(test_ids).index(i) for i in ptids]

    with open('./data/ccs_codes_all_item_categories.pickle', 'rb') as f:
        item_cats = pickle.load(f)
    f.close()
    itemdict = item_cats['cat'].to_dict()

    with open('./results/example_pts_weights.pickle', 'rb') as f:
        seq_wts, code_wts = pickle.load(f)

    all_wts = []
    for x, p in enumerate(inds):
        meds_seq = get_codes(test, p, itemdict)
        wts = aggregate_code_wts(meds_seq, code_wts[x])
        all_wts.append(wts)

    exmple_pos = result2[result2['ptid'].isin(ptids1)]
    exmple_neg = result_neg[result_neg['ptid'].isin(ptids0)]

    # get demographics info:
    with open('./data/orders_pt_info.pickle', 'rb') as f:
        pt_info_orders = pickle.load(f)

    ages = pt_info_orders[['ptid', 'age']].drop_duplicates().groupby('ptid').min()
    ages = ages['age'].to_dict()

    ages_exmple = [ages[x] for x in ptids]
    actual_ages = [77, 71, 81, 51, 64, 66, 44, 12, 69, 55, 66, 12]
    ptids1 = ['1446009', '1150550', '554938', '1348961', '1539136', '1703959']
    ptids0 = ['1790610', '1634904', '1218594', '773135', '1334050', '1596720']

    # selected pts for further analysis
    ptids = ['1446009', '1539136', '1703959']
    inds = [list(test_ids).index(i) for i in ptids]
    # ages_exmple = [ages[x] for x in ptids]
    exmple = result2[result2['ptid'].isin(ptids)]
    exmple_seq = [test[i] for i in inds]
    seq_wts_exmple = [seq_wts[i] for i in [0, 4, 5]]
    code_wts_exmple = [all_wts[i] for i in [0, 4, 5]]

    with open('./results/example_pts_info.pickle', 'wb') as f:
        pickle.dump([exmple, exmple_seq, seq_wts_exmple, code_wts_exmple], f)

    ages = [77, 64, 66]  # 10 year is 0.443
    testdemoip_exm_new1 = [[0.0, 1.4008357307722108, 1.0], [1.0, 1.4008357307722108, 1.0],
                          [0.0, 1.4008357307722108 + 0.443, 1.0], [0.0, 1.4008357307722108, 0.0]]
    testdemoip_exm_new2 = [[0.0, 0.8249648802368135, 0.0], [1.0, 0.8249648802368135, 0.0],
                           [0.0, 0.8249648802368135 + 0.443, 0.0], [0.0, 0.8249648802368135, 1.0]]
    testdemoip_exm_new3 = [[1.0, 0.9135603957037977, 0.0], [0.0, 0.9135603957037977, 0.0],
                           [1.0, 0.9135603957037977 + 0.443, 0.0], [1.0, 0.9135603957037977, 1.0]]

    new_pred_vals = [0.9640013575553894,
                     0.955817461013794,
                     0.9713656902313232,
                     0.9620007276535034,
                     0.7463871240615845,
                     0.7039254903793335,
                     0.7885023355484009,
                     0.7568705081939697,
                     0.7960509657859802,
                     0.8285189270973206,
                     0.8317776918411255,
                     0.8050177097320557]

    # ============================= List of top 10 most important items in hospitalized pts ===============
    # get the code wts of all testing pts
    model_type = 'crnn2-bi-tanh-fn'
    output_file = './results/test_outputs_pos' + model_type + '_layer1.pickle'
    with open(output_file, 'rb') as f:
        pos_ids, seq_wts_pos, code_wts_pos, context_pos, demoips_pos, test_pos = pickle.load(f)
    f.close()

    with open('./data/hospitalization_train_validate_test_ids.pickle', 'rb') as f:
        train_ids, valid_ids, test_ids = pickle.load(f)

    with open('./data/ccs_codes_all_item_categories.pickle', 'rb') as f:
        item_cats = pickle.load(f)
    f.close()
    itemdict = item_cats['cat'].to_dict()


    def aggregate_code_wts_items_top(meds_seq, seq_wts, data, n):
        w = {}
        for i in range(4):
            if len(meds_seq[i]) > 0:
                for k in range(len(meds_seq[i])):
                    if not w.__contains__(meds_seq[i][k]):
                        w[meds_seq[i][k]] = 0
                    w[meds_seq[i][k]] += data[i][k] * seq_wts[i]
        if w.__contains__('p0'):
            del w['p0']
        if w.__contains__('dx259'):
            del w['dx259']
        wk = list(w.keys())
        wv = list(w.values())
        wv = normalize(wv, norm='l1').tolist()[0]
        w1 = dict(zip(wk, wv))
        wts = sorted(w1.items(), key=operator.itemgetter(1), reverse=True)[:n]
        # wts = w1.items().sort(key=operator.itemgetter(1), reverse=True)
        items = [j[0] for j in wts]
        return items, len(w)

    def get_top_items(inds, test_pos, itemdict, seq_wts_pos, code_wts_pos, n1, n2):
        top_items = []
        num_items = []
        for x in inds:
            meds_seq = get_codes(test_pos, x, itemdict)
            items, nums = aggregate_code_wts_items_top(meds_seq, seq_wts_pos[x], code_wts_pos[x], n=n1)
            top_items += items
            num_items.append(nums)
        top_items_cts = dict(Counter(top_items))
        top_items_sorted = sorted(top_items_cts.items(), key=operator.itemgetter(1), reverse=True)[:n2]
        return top_items_sorted

    top_items_sorted = get_top_items(range(len(test_pos)), test_pos, itemdict, seq_wts_pos, code_wts_pos, 5, 30)
    # top_items_sorted
    # [('p33', 834),
    #  ('p171', 772),
    #  ('p19', 674),
    #  ('p233', 532),
    #  ('Diagnostic Products', 384),
    #  ('p235', 370),
    #  ('dx98', 352),
    #  ('dx211', 283),
    #  ('dx205', 278),
    #  ('p112', 267),
    #  ('dx133', 258),
    #  ('p61', 256),
    #  ('dx53', 241),
    #  ('dx257', 226),
    #  ('Analgesics-Narcotic', 224),
    #  ('dx49', 222),
    #  ('dx663', 214),
    #  ('p211', 209),
    #  ('dx95', 194),
    #  ('dx258', 190)]

    # =========================== visualize the learned representation for patients in some groups =========
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

    def process_pdxs(data, dxgrps_dict, dxgrps_dict2):
        data['pdx_ip'] = data['pdx_ip'].str.replace('.', '')
        dxs = data['pdx_ip'].values.tolist()
        dxcats = []
        for i in dxs:
            if dxgrps_dict.__contains__(i):
                dxcats.append(dxgrps_dict[i])
            elif dxgrps_dict2.__contains__(i):
                dxcats.append(dxgrps_dict2[i])
            else:
                dxcats.append('0')
        data['pdxcat'] = dxcats
        data = data[data['pdxcat'] != '0']
        return data

    result2_repeated = result2[['ptid', 'pdx_ip']].groupby('ptid').count()
    result2_repeated.reset_index(inplace=True)
    result2_repeated = result2_repeated[result2_repeated['pdx_ip'] > 1]
    repeat_ids = list(result2_repeated['ptid'].values)

    dxgrps0, dxgrps_dict0, dxgrps_dict20 = dx2dxcat()
    result3 = result2[~result2['ptid'].isin(repeat_ids)]
    result3 = process_pdxs(result3, dxgrps_dict0, dxgrps_dict20)
    result3_pdxgrp_cts = result3[['ptid', 'pdxcat']].groupby('pdxcat').count()
    result3_pdxgrp_cts.reset_index(inplace=True)
    result3_pdxgrp_cts = result3_pdxgrp_cts.sort(['ptid'], ascending=[0])

    pos_203 = result3[result3['pdxcat'] == '203']
    pos_2 = result3[result3['pdxcat'] == '2']
    pos_100 = result3[result3['pdxcat'] == '100']
    pos_108 = result3[result3['pdxcat'] == '108']
    pos_50 = result3[result3['pdxcat'] == '50']
    ids_203 = list(pos_203['ptid'].values)
    ids_2 = list(pos_2['ptid'].values)
    ids_100 = list(pos_100['ptid'].values)
    ids_108 = list(pos_108['ptid'].values)
    ids_50 = list(pos_50['ptid'].values)
    inds_203 = [pos_ids.index(i) for i in ids_203]
    inds_2 = [pos_ids.index(i) for i in ids_2]
    inds_100 = [pos_ids.index(i) for i in ids_100]
    inds_108 = [pos_ids.index(i) for i in ids_108]
    inds_50 = [pos_ids.index(i) for i in ids_50]
    top_items_203 = get_top_items(inds_203, test_pos, itemdict, seq_wts_pos, code_wts_pos, 5, 30)
    top_items_2 = get_top_items(inds_2, test_pos, itemdict, seq_wts_pos, code_wts_pos, 5, 30)
    top_items_100 = get_top_items(inds_100, test_pos, itemdict, seq_wts_pos, code_wts_pos, 5, 30)
    top_items_108 = get_top_items(inds_108, test_pos, itemdict, seq_wts_pos, code_wts_pos, 5, 30)
    top_items_50 = get_top_items(inds_50, test_pos, itemdict, seq_wts_pos, code_wts_pos, 5, 30)
    # # visualize in t-sne
    # ys = [1] * len(pos_203) + [2] * len(pos_2) + [3] * len(pos_100) + [4] * len(pos_108)

    #
    # import numpy as np
    # context_pos_all = np.concatenate((context_pos, demoips_pos), axis=1)
    # context_groups = [context_pos_all[pos_ids.index(i)] for i in ids]
    #
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # tsne = TSNE(n_components=2, n_iter=5000)
    # context_2d = tsne.fit_transform(context_groups)
    # contex_x = [a[0] for a in context_2d]
    # contex_y = [a[1] for a in context_2d]
    # df = pd.DataFrame(dict(x=contex_x, y=contex_y, color=ys))
    # sns.lmplot('x', 'y', data=df, hue='color', fit_reg=False)
    # plt.show()

