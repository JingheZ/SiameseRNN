"""
Define a cutoff, observation window (one-year) and prediction window (one-year)
"""

import pickle
import pandas as pd


def process_doc_data_sep(filenames, vids, vlen):
    # load data
    with open('./data/visit_ranks.pickle', 'rb') as f:
        visit_ranks_reverse, _ = pickle.load(f)
    f.close()

    with open(filenames[0], 'rb') as f:
        data_dx = pickle.load(f)
    f.close()

    with open(filenames[1], 'rb') as f:
        data_med = pickle.load(f)
    f.close()

    with open(filenames[2], 'rb') as f:
        data_proc = pickle.load(f)
    f.close()

    vdoc_dx = get_items_in_visit(data_dx, visit_ranks_reverse, vids, vlen)
    vdoc_med = get_items_in_visit(data_med, visit_ranks_reverse, vids, vlen)
    vdoc_proc = get_items_in_visit(data_proc, visit_ranks_reverse, vids, vlen)

    with open('./data/hospitalization_1st_year_visit_dxs.pickle', 'wb') as f:
        pickle.dump(vdoc_dx, f)
    f.close()
    with open('./data/hospitalization_1st_year_visit_meds.pickle', 'wb') as f:
        pickle.dump(vdoc_med, f)
    f.close()
    with open('./data/hospitalization_1st_year_visit_procs.pickle', 'wb') as f:
        pickle.dump(vdoc_proc, f)
    f.close()
    # return vdoc_dx, vdoc_med, vdoc_proc


def get_items_in_visit(data0, doc, vids, vlen):
    data = data0[data0['vid'].isin(vids)]
    vdoc = {}
    for i in data.index:
        pid = data['ptid'].loc[i]
        vid = data['vid'].loc[i]
        item = data['itemid'].loc[i]
        rank = doc[pid][vid]
        if not vdoc.__contains__(pid):
            length = vlen['rank'].loc[pid] + 1
            vdoc[pid] = [[] for x in range(length)]
        vdoc[pid][rank].append(str(item))
    return vdoc


def get_visit_type(data, vlen):
    vtype = {}
    for i in data.index:
        pid = data['ptid'].loc[i]
        rank = data['rank'].loc[i]
        tp = data['cdrIPorOP'].loc[i]
        if not vtype.__contains__(pid):
            length = vlen['rank'].loc[pid] + 1
            vtype[pid] = [[] for x in range(length)]
        vtype[pid][rank].append(tp)
    return vtype


def get_pt_demographics(ptids):
    with open('./data/orders_pt_info.pickle', 'rb') as f:
        pt_info = pickle.load(f)
    f.close()
    pt_info2 = pt_info[pt_info['ptid'].isin(ptids)]
    pt_info2['gender'] = pt_info2['sex'].map({'M': 0, 'F': 1})
    pt_info2.to_csv('./data/hospitalization_1st_year_pt_demographics.csv', index=False)


if __name__ == '__main__':
    with open('./data/visits_v4.pickle', 'rb') as f:
        visits = pickle.load(f)
    f.close()
    visits['rank'] = visits['rank'].astype(int)
    # select patients with at least 2 visits in first year and have a total length at least two years
    visits['adm_day'] = visits['anon_adm_date_y'] / 60 / 24
    visits['dis_day'] = visits['anon_dis_date_y'] / 60 / 24
    max_adm = visits[['ptid', 'adm_day']].groupby('ptid').max()
    max_adm.describe()
    max_dis = visits[['ptid', 'dis_day']].groupby('ptid').max()
    max_dis.describe()
    ptids_adm = max_adm[max_adm['adm_day'] > 365].index # 117909 pt with last visit at least after 1 year of first visit
    ptids_dis = max_dis[max_dis['dis_day'] > 730].index # 86155 pt
    ptids = set(ptids_adm).intersection(set(ptids_dis)) # 85087 pt
    visits_2yr = visits[visits['ptid'].isin(ptids)]
    visits_2yr = visits_2yr[visits_2yr['adm_day'] <= 365]
    counts = visits_2yr[['ptid', 'rank']].groupby('ptid').count()
    counts2_ptids = counts[counts['rank'] >= 1].index     # at least one visits in first year: 85087 pts
    ptids = counts2_ptids

    # select patients with at least one visits in the first year and a IP visit in the next years
    visit_ip = visits[visits['cdrIPorOP'] == 'IP'] # 57837 pts
    visit_ip = visit_ip[visit_ip['adm_day'] > 365] # 13010 pts
    # negative class pts
    ptids_nonip = set(ptids).difference(set(visit_ip['ptid'].values)) # 60070 pts
    # positive class pts
    visit_ip = visit_ip[visit_ip['rank'] >= 1]  # 13010 pts
    visit_ip = visit_ip[visit_ip['adm_day'] <= 730] # 6338 patients
    ptids_ip = set(visit_ip['ptid'].values)

    ptids_both = set(ptids).union(ptids_ip) # 87003 pts
    with open('./data/hospitalization_both_pos_neg_ptids.pickle', 'wb') as f:
        pickle.dump([list(ptids_both), list(ptids_ip), list(ptids_nonip)], f)
    f.close()

    # get the first year data for training
    visits2 = visits[visits['ptid'].isin(ptids_both)]
    visits2_1yr = visits2[visits2['adm_day'] <= 365]
    with open('./data/hospitalization_1st_year_data.pickle', 'wb') as f:
        pickle.dump(visits2_1yr, f)
    f.close()

    # get the visit type
    visits2_1yr_type = visits2_1yr[['ptid', 'rank', 'cdrIPorOP']].drop_duplicates()
    visits2_1yr_type = visits2_1yr_type.sort(['ptid', 'rank'])
    visit_lengths = visits2_1yr_type[['ptid', 'rank']].groupby('ptid').max()
    with open('./data/hospitalization_1st_year_visit_lengths.pickle', 'wb') as f:
        pickle.dump(visit_lengths, f)
    f.close()
    visit_types = get_visit_type(visits2_1yr_type, visit_lengths)
    with open('./data/hospitalization_1st_year_visit_types.pickle', 'wb') as f:
        pickle.dump(visit_types, f)
    f.close()

    # get patient demographics
    get_pt_demographics(ptids_both)

    # get the dx, med, proc of these visits separately
    vids = set(visits2_1yr['vid'].values)
    filenames = ['./data/dxs_data_v2.pickle', './data/med_orders_v2.pickle', './data/proc_orders_v2.pickle']
    process_doc_data_sep(filenames, vids, visit_lengths)

