__author__ = 'jinghe'

'''
select the patients data according to the patient ids from both med and procedure datasets
'''

"""import packages"""
import pickle
import pandas as pd


def load_data(filename, colnames):
    data = pd.read_csv(filename, delimiter='\t', header=None)
    data.columns = colnames
    data['ptid'] = data['ptid'].astype(str)
    data['vid'] = data['vid'].astype(str)
    return data


def count_stat_summary(data):
    # num_pts = len(data['ptid'].unique()) # 379968 pts
    # avg_num_visits = data[['ptid', 'vid']].drop_duplicates().groupby(['ptid']).count().mean() # average 4.31 visits per patient, median: 2
    # num_dxs = len(data['dx'].unique()) # 11259 codes
    # num_primary_dxs = len(data['primary_dx'].unique())  # 8661 primary codes
    # avg_num_codes = data[['ptid', 'dx']].drop_duplicates().groupby(['ptid']).count().mean() # 11.32 dx per patient, median: 5.0
    # avg_num_codes_by_v = data[['ptid', 'vid', 'dx']].groupby(['ptid', 'vid']).count().mean() # 4.1 dx per patient per visit, median:3
    # avg_num_primary_codes = data[['ptid', 'primary_dx']].drop_duplicates().groupby(['ptid']).count().mean() # 2.92 dx per patient, median: 2.0
    visits = data['vid'].unique() # 1638104 visits
    return visits


def process_orders(filename, colnames, colnames_selected, item_colname):
    data = load_data(filename, colnames)
    data = data[colnames_selected]
    # all_items = []
    if len(item_colname) > 1:
        all_items = data[item_colname].drop_duplicates()
    else:
        all_items = data[item_colname[0]].unique()
    visits = data['vid'].unique()
    return data, visits, all_items


def get_orders_visit_info(data_med, data_proc):
    visit_info_med = data_med[['ptid', 'vid', 'anon_adm_date', 'anon_dis_date', 'VisitDXs', 'cdrIPorOP', 'timeline']]
    visit_info_proc = data_proc[['ptid', 'vid', 'anon_adm_date', 'anon_dis_date', 'VisitDXs', 'cdrIPorOP', 'timeline']]
    visit_info_orders = pd.concat((visit_info_med, visit_info_proc), axis=0)
    visit_info_orders = visit_info_orders.drop_duplicates()
    with open('./data/orders_visit_info.pickle', 'wb') as f:
        pickle.dump(visit_info_orders, f)
    f.close()
    # return visit_info_orders


def get_orders_pt_info(data_med, data_proc):
    pt_info_med = data_med[['ptid', 'age', 'sex']]
    pt_info_proc = data_proc[['ptid', 'age', 'sex']]
    pt_info_orders = pd.concat((pt_info_med, pt_info_proc), axis=0)
    pt_info_orders = pt_info_orders.drop_duplicates()
    with open('./data/orders_pt_info.pickle', 'wb') as f:
        pickle.dump(pt_info_orders, f)
    f.close()
    # return pt_info_orders


def create_visit_ranks(data):
    # create a dict for each patient, key is the rank of the visit, value are the vids
    visit_ranks = {}
    previous_ptid = ''
    j = 0
    previous_adm_time = 0
    for i in data.index:
        ptid = data['ptid'].loc[i]
        adm_time = data['anon_adm_date_y'].loc[i]
        if ptid != previous_ptid:
            j = 0
            visit_ranks[ptid] = {0: []}
            previous_adm_time = adm_time
        if adm_time != previous_adm_time:
            j += 1
            visit_ranks[ptid][j] = []
        visit_ranks[ptid][j].append(data['vid'].loc[i])
        previous_adm_time = adm_time
        previous_ptid = ptid
    return visit_ranks


def create_visit_ranks_reverse(visit_ranks):
    # reverse the dict in visit_rank, the key-value are exchanged
    visit_ranks_r = {}
    visit_ranks_r_flatten = {}
    for k, v in visit_ranks.items():
        visit_ranks_r[k] = {}
        for k0, v0 in v.items():
            for v00 in v0:
                visit_ranks_r[k][v00] = k0
                visit_ranks_r_flatten[v00] = k0
    return visit_ranks_r, visit_ranks_r_flatten


if __name__ == '__main__':
    # ============================ DX Data =================================================
    filename = './data/2016_04apr_13_ALL_Diagnoses.dat'
    data_dx = load_data(filename, colnames=['ptid', 'vid', 'dx', 'primary_dx'])
    all_dxs = data_dx['dx'].unique()
    visits_dx = count_stat_summary(data_dx)

    # ============================ Med Data ===================================================
    filename = './data/2016_04apr_13_Orders_Meds_redo.dat'
    colnames_med = ['med_order_num', 'Med_Nme', 'Med_Thera_Cls', 'Med_Pharm_Cls', 'Med_Pharm_Sub_Cls',
                    'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'anon_adm_date', 'anon_dis_date', 'timeline',
                    'cdrIPorOP', 'order_pt_loc', 'author_physid_deid', 'order_physid_deid']
    selected_colnames_med = ['ptid', 'age', 'sex', 'vid', 'Med_Pharm_Cls', 'VisitDXs', 'timeline', 'anon_adm_date',
                         'anon_dis_date', 'cdrIPorOP']
    data_med, visit_med, all_meds = process_orders(filename, colnames_med, selected_colnames_med, ['Med_Pharm_Cls'])

    # ============================ Procedure Data ===================================================
    filename = './data/2016_04apr_13_Orders_Procedures_redo.dat'
    colnames_proc = ['proc_order_num', 'PROC_ID', 'PROC_NAME', 'PROC_CAT', 'PROC_CAT_ID', 'ORDER_DISPLAY_NAME',
                    'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date', 'anon_dis_date',
                    'cdrIPorOP', 'physid_deid']
    selected_colnames_proc = ['PROC_ID', 'PROC_NAME', 'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline',
                             'anon_adm_date', 'anon_dis_date', 'cdrIPorOP']
    data_proc, visit_proc, all_procs = process_orders(filename, colnames_proc, selected_colnames_proc, ['PROC_ID', 'PROC_NAME'])



    with open('./data/all_dx_meds_procs.pickle', 'wb') as f:
        pickle.dump([all_dxs, all_meds, all_procs], f)
    f.close()

    # ============================= Analyze pts and visits in orders and dx data ==============================
    get_orders_visit_info(data_med, data_proc)
    get_orders_pt_info(data_med, data_proc)

    vid_orders = set(visit_proc.tolist()).union(set(visit_med.tolist()))
    vid_diff_order2dx = vid_orders.difference(set(visits_dx.tolist()))
    vid_diff_dx2order = set(visits_dx.tolist()).difference(vid_orders)
    print('Number of distinct visits in dx data: %i' % len(visits_dx))
    print('Number of distinct visits in order data: %i' % len(vid_orders))
    print('Number of distinct visits in order data but not in dx data: %i' % len(vid_diff_order2dx))
    print('Number of distinct visits in dx data but not in order data: %i' % len(vid_diff_dx2order))
    with open('./data/visits_not_in_dxs.pickle', 'wb') as f:
        pickle.dump(vid_diff_order2dx, f)
    f.close()

    # ============================== Save data to pickle ========================================================
    with open('./data/dxs_data.pickle', 'wb') as f:
        pickle.dump(data_dx, f)
    f.close()
    data_dx.to_csv('./data/dxs_data.csv', index=False)

    with open('./data/med_orders.pickle', 'wb') as f:
        pickle.dump(data_med[['ptid', 'vid', 'Med_Pharm_Cls', 'anon_adm_date']], f)
    f.close()
    data_med[['ptid', 'vid', 'Med_Pharm_Cls', 'anon_adm_date']].to_csv('./data/med_orders.csv', index=False)

    with open('./data/proc_orders.pickle', 'wb') as f:
        pickle.dump(data_proc[['ptid', 'vid', 'PROC_ID', 'anon_adm_date']], f)
    f.close()
    data_proc[['ptid', 'vid', 'PROC_ID', 'anon_adm_date']].to_csv('./data/proc_orders.csv', index=False)
    print('Done!')

    # ================================ load data back and analyze the visit in orders but not appearing in dxs ======
    with open('./data/dxs_data.pickle', 'rb') as f:
        data_dx = pickle.load(f)
    f.close()
    dxs_visits = data_dx[['ptid', 'vid']].drop_duplicates()

    with open('./data/orders_visit_info.pickle', 'rb') as f:
        visit_info_orders = pickle.load(f)
    f.close()

    with open('./data/visits_not_in_dxs.pickle', 'rb') as f:
        vid_diff_order2dx = pickle.load(f)
    f.close()

    orders_visits = visit_info_orders[['ptid', 'vid']].drop_duplicates()

    visits_not_in_dxs = orders_visits[orders_visits['vid'].isin(vid_diff_order2dx)] # 1661 rows with vid and ptid, where these vid not appeared in dxs data
    pts_with_visits_not_in_dxs = set(visits_not_in_dxs['ptid'].values) # 1201 patients with visits not appeared in dxs data
    orders_pts_with_visits_not_in_dxs = orders_visits[orders_visits['ptid'].isin(pts_with_visits_not_in_dxs)]
    # 1661 visits, hence all these 1201 patients have no other records with dxs info in the dx data;
    # so just need to remove these 1201 patients from med and proc order data

    visit_info_orders_updated = visit_info_orders[~visit_info_orders['ptid'].isin(pts_with_visits_not_in_dxs)]
    with open('./data/orders_visit_info_updated.pickle', 'wb') as f:
        pickle.dump(visit_info_orders_updated, f)
    f.close()

    # ================================ Analyze sequence lengths and how many visits before hospitalization ==========
    with open('./data/orders_visit_info_updated.pickle', 'rb') as f:
        visit_info_orders_updated = pickle.load(f)
    f.close()
    del visit_info_orders_updated['timeline']
    # remove patients with negative visit discharge
    visits = visit_info_orders_updated
    pts_negative_disch = visits[visits['anon_dis_date'] < 0]
    pts_negative_disch = set(pts_negative_disch['ptid'].values) # remove 81489 patients with negative discharge time
    visits = visits[~visits['ptid'].isin(pts_negative_disch)].drop_duplicates() # 298,479 patients
    visits = visits.sort(['ptid', 'vid', 'anon_adm_date']).drop_duplicates()
    pt_ids = list(set(visits['ptid'].values)) # 298,479 patients

    # process the admission time and visit id:
    # OP visits with the same ids, give the earliest admission time to that visit;
    # then, visits with different ids but same times are used as same visit
    visit_adm_time = visits[['vid', 'anon_adm_date']].groupby('vid').min()
    visit_adm_time['vid'] = visit_adm_time.index
    visits_v2 = pd.merge(left=visits, right=visit_adm_time, how='inner', left_on='vid', right_on='vid')
    del visits_v2['anon_adm_date_x']
    del visits_v2['anon_dis_date']
    del visits_v2['VisitDXs']
    visits_v2['cdrIPorOP'] = visits_v2['cdrIPorOP'].map({'OP': 'OP', 'IP': 'IP', 'OBS': 'OP'})
    visits_v2 = visits_v2.sort(['ptid', 'anon_adm_date_y', 'vid']).drop_duplicates()

    # select pts with at least two visits
    counts = visits_v2[['ptid', 'anon_adm_date_y']].drop_duplicates().groupby('ptid').count() # 298479 visits
    counts2 = counts[counts['anon_adm_date_y'] >= 2] # 156,224 patients
    visits_v3 = visits_v2[visits_v2['ptid'].isin(counts2.index)].drop_duplicates()
    visits_v3 = visits_v3.sort(['ptid', 'anon_adm_date_y'])
    # duplicated visits which each visit has more than one visit, take it as an inpatient visit
    vids_multiple_visit_type = set(visits_v3[visits_v3.duplicated('vid')]['vid'])
    visits_v3.ix[visits_v3['vid'].isin(vids_multiple_visit_type), 'cdrIPorOP'] = 'IP'
    visits_v3 = visits_v3.drop_duplicates()
    with open('./data/visits_v3.pickle', 'wb') as f:
        pickle.dump(visits_v3, f)
    f.close()
    # create a dict of all patients, in each patient data, there is also a dict of ranks containing the visit ids
    visit_ranks_dict = create_visit_ranks(visits_v3)
    # create dict of dict for patient, that the keys are vid and value is rank
    visit_ranks_reverse, visit_ranks_reverse_flattened = create_visit_ranks_reverse(visit_ranks_dict)
    with open('./data/visit_ranks.pickle', 'wb') as f:
        pickle.dump([visit_ranks_reverse, visit_ranks_reverse_flattened], f)
    f.close()

    # create a dict of all patients which flattens the above dict
    visit_ranks_df = pd.DataFrame.from_dict(visit_ranks_reverse_flattened, dtype=object, orient='index')
    visit_ranks_df['vid'] = visit_ranks_df.index
    visit_ranks_df.columns = ['rank', 'vid']
    visits_v4 = pd.merge(left=visits_v3, right=visit_ranks_df, how='inner', left_on='vid', right_on='vid')
    visits_v4 = visits_v4.sort(['ptid', 'rank'])
    with open('./data/visits_v4.pickle', 'wb') as f:
        pickle.dump(visits_v4, f)
    f.close()

    IPvisits = visits_v4[visits_v4['cdrIPorOP'] == 'IP'] # 45995 patients with inhospital visits
    IPvisits2 = IPvisits[IPvisits['rank'] > 1]  # 19406 patients with inhospital visits after two visits
    IPvisits1 = IPvisits[IPvisits['rank'] > 0]  # 26907 patients with inhospital visits after the first visits
    IPvisits0 = IPvisits[IPvisits['rank'] == 0]  # 28106 patients with inhospital visits is the first visit
    IPvisits_only0 = IPvisits0[~IPvisits0['ptid'].isin(set(IPvisits1['ptid'].values))] # 19381 patients with inpatient visits in the first visit

    visits_v5 = visits_v4[['ptid', 'rank', 'anon_adm_date_y']]


    # ====================== Remove the unused pts from the orders and dx data =====================
    ptids = list(visit_ranks_reverse.keys())
    data_dx.columns = ['ptid', 'vid', 'itemid', 'pdx']
    data_dx = data_dx[data_dx['ptid'].isin(ptids)]

    data_med.columns = ['ptid', 'vid', 'itemid', 'adm_date']
    data_med = data_med[data_med['ptid'].isin(ptids)]

    data_proc.columns = ['ptid', 'vid', 'itemid', 'adm_date']
    data_proc = data_proc[data_proc['ptid'].isin(ptids)]

    with open('./data/dxs_data_v2.pickle', 'wb') as f:
        pickle.dump(data_dx, f)
    f.close()
    data_dx.to_csv('./data/dxs_data_v2.csv', index=False)

    with open('./data/med_orders_v2.pickle', 'wb') as f:
        pickle.dump(data_med[['ptid', 'vid', 'itemid']], f)
    f.close()
    data_med[['ptid', 'vid', 'itemid']].to_csv('./data/med_orders_v2.csv', index=False)

    with open('./data/proc_orders_v2.pickle', 'wb') as f:
        pickle.dump(data_proc[['ptid', 'vid', 'itemid']], f)
    f.close()
    data_proc[['ptid', 'vid', 'itemid']].to_csv('./data/proc_orders_v2.csv', index=False)



    # hence, if predict hospialization with at least one visit as prior info, there are 26907 patients,
    # and 156224 - 26907 pts in negative class
    # But need to work on the time window, to exclude some info before the prediction window
    # also need to study the prediction window and how to structure it as a semi-supervised learning method

    # To do:
    # 1. extract the dxs, meds, procedures to learn a vector representation
    # need tp think about how to structure it since it is two-level: pt level and visit level
    # think about GloVe which uses co-occurrence counts rather than orders,
    # There are orders between visits, and no orders within visits
    # One solution: make each visit as a document to learn code vectors, do not consider patients
    # The other solution is consider the visits by patients


