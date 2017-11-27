__author__ = 'jinghe'

'''
select the patients data according to the patient ids from both med and procedure datasets
'''

"""import packages"""
import pickle
import pandas as pd


def load_data(filename, colnames):
    data = pd.read_csv(filename, delimiter='\t', header=None, encoding="ISO-8859-1")
    data.columns = colnames
    data['ptid'] = data['ptid'].astype(str)
    data['vid'] = data['vid'].astype(str)
    return data


def count_stat_summary(data):
    # num_pts = len(data['ptid'].unique()) # 473915 pts
    # avg_num_visits = data[['ptid', 'vid']].drop_duplicates().groupby(['ptid']).count().mean() # average 4.95 visits per patient, median: 2
    # num_dxs = len(data['dx'].unique()) # 34419 codes
    # num_primary_dxs = len(data['primary_dx'].unique())  # 22523 primary codes
    # avg_num_codes = data[['ptid', 'dx']].drop_duplicates().groupby(['ptid']).count().mean() # 13.19 dx per patient, median: 6.0
    # avg_num_codes_by_v = data[['ptid', 'vid', 'dx']].groupby(['ptid', 'vid']).count().mean() # 4.1 dx per patient per visit, median:3
    # avg_num_primary_codes = data[['ptid', 'primary_dx']].drop_duplicates().groupby(['ptid']).count().mean() # 3.37 dx per patient, median: 2.0
    visits = data['vid'].unique() # 2343651 visits
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


def proc2proccat(all_procs):
    filename = './data/2017_ccs_services_procedures.csv'
    # sed - i 's/,//g'. / data / 2017_ccs_services_procedures.csv
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
        left = data['icd'].apply(lambda x: x.split('-')[0])
        right = data['icd'].apply(lambda x: x.split('-')[1])
        data['left'] = left
        data['right'] = right
        proc_grps = data[['left', 'right', 'category']]
        return proc_grps

    procgrps = CCS(filename)
    procgrps = procgrps.values.tolist()
    cats = []
    all_procs['PROC_ID'] = all_procs['PROC_ID'].astype(str)
    for i in all_procs.index:
        cat = '0'
        proc = all_procs['PROC_ID'].loc[i]
        for l, r, c in procgrps:
            if proc >= l and proc <= r:
                cat = c
                break
        cats.append(cat)
    all_procs['proccat'] = cats
    proc_data = all_procs[['PROC_ID', 'proccat']]
    proc_data['PROC_ID'] = proc_data['PROC_ID'].apply(lambda x: 'p' + x)
    # proc_data.index = proc_data['PROC_ID']
    # del proc_data['PROC_ID']
    return procgrps, all_procs, proc_data


def process_pdxs(data, dxgrps_dict, dxgrps_dict2):
    data['pdx'] = data['pdx'].str.replace('.', '')
    dxs = data['pdx'].values.tolist()
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
    df = df[['ptid', 'itemid']].drop_duplicates()
    counts = df.groupby(['ptid', 'itemid']).size().unstack('itemid').fillna(0)
    counts = filter_rare_columns(counts, thres)
    counts['response'] = y
    return counts


def get_counts_subwindow(df, y, vars, c):
    def get_counts_one_window(counts0, j):
        cts = counts0[counts0['subw'] == j]
        del cts['subw']
        cts.columns = ['ptid'] + ['t' + str(j) + '_' + k for k in cts.columns[1:]]
        return cts
    # add IP or OP indicator
    df_ip = df[df['cdrIPorOP'] == 'IP']
    ip_ids = set(df_ip['ptid'].values)

    df = df[df['itemid'].isin(vars)]
    df['subw'] = df['adm_month'].apply(lambda x: int(x / c))
    counts0 = df[['ptid', 'itemid', 'subw']].groupby(['ptid', 'itemid', 'subw']).size().unstack('itemid')
    counts0.reset_index(inplace=True)
    dt = get_counts_one_window(counts0, 0)
    for j in range(1, int(12/c)):
        cts = get_counts_one_window(counts0, j)
        dt = pd.merge(dt, cts, on='ptid', how='outer')

    dt['IPvisit'] = dt['ptid'].apply(lambda x: 1 if x in ip_ids else 0)
    # add response
    dt['response'] = y
    dt.index = dt['ptid']
    del dt['ptid']
    dt.fillna(0, inplace=True)
    return dt


def process_dxs(dxs, dxgrps_dict, dxgrps_dict2):
    dxs = [i.replace('.', '') for i in dxs]
    dxcats = []
    for i in dxs:
        if dxgrps_dict.__contains__(i):
            dxcats.append(dxgrps_dict[i])
        elif dxgrps_dict2.__contains__(i):
            dxcats.append(dxgrps_dict2[i])
        else:
            dxcats.append('0')
    dxs_df = pd.DataFrame([dxs, dxcats])
    dxs_df = dxs_df.transpose()
    dxs_df.columns = ['dx', 'dxcat']
    dxs_df['dx'] = dxs_df['dx'].apply(lambda x: 'dx' + x)
    # dxs_df.index = dxs_df['dx']
    # del dxs_df['dx']
    return dxs_df


def get_all_item_categories(dx_df, proc_df, all_meds):
    # dx_df.reset_index(inplace=True)
    dx_df.columns = ['code', 'cat']
    dx_df['cat'] = dx_df['cat'].apply(lambda x: 'dx' + x)
    # proc_df.reset_index(inplace=True)
    proc_df.columns = ['code', 'cat']
    proc_df['cat'] = proc_df['cat'].apply(lambda x: 'p' + x)
    med_df = pd.DataFrame([all_meds.tolist(), all_meds.tolist()])
    med_df = med_df.transpose()
    med_df.columns = ['code', 'cat']
    item_cats = pd.concat([dx_df, med_df, proc_df], axis=0)
    item_cats.index = item_cats['code']
    del item_cats['code']
    return item_cats


if __name__ == '__main__':
    # ============================ DX Data =================================================
    filename = './data/2017_02feb_28_ALL_Diagnoses.dat'
    data_dx = load_data(filename, colnames=['ptid', 'vid', 'dx', 'primary_dx'])
    all_dxs = data_dx['dx'].unique()
    visits_dx = count_stat_summary(data_dx)

    # ============================ Med Data ===================================================

    filename = './data/2017_02feb_28_Orders_Meds.dat'
    colnames_med = ['med_order_num', 'Med_Nme', 'Med_Thera_Cls', 'Med_Pharm_Cls', 'Med_Pharm_Sub_Cls',
                    'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date', 'anon_dis_date',
                    'cdrIPorOP', 'order_pt_loc', 'author_physid_deid', 'order_physid_deid']
    selected_colnames_med = ['ptid', 'age', 'sex', 'vid', 'Med_Pharm_Cls', 'VisitDXs', 'timeline', 'anon_adm_date',
                         'anon_dis_date', 'cdrIPorOP']
    data_med, visit_med, all_meds = process_orders(filename, colnames_med, selected_colnames_med, ['Med_Pharm_Cls'])
    with open('./data/all_dx_and_meds.pickle', 'wb') as f:
        pickle.dump([all_dxs, all_meds], f)
    f.close()
    # ============================ Procedure Data ===================================================
    filename = './data/2017_02feb_28_Orders_Procedures.dat'
    colnames_proc = ['proc_order_num', 'PROC_ID', 'PROC_NAME', 'PROC_CAT', 'PROC_CAT_ID', 'ORDER_DISPLAY_NAME',
                    'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date', 'anon_dis_date',
                    'cdrIPorOP', 'physid_deid']
    selected_colnames_proc = ['PROC_ID', 'PROC_NAME', 'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline',
                             'anon_adm_date', 'anon_dis_date', 'cdrIPorOP']
    data_proc, visit_proc, all_procs = process_orders(filename, colnames_proc, selected_colnames_proc, ['PROC_ID', 'PROC_NAME'])

    with open('./data/all_procs.pickle', 'wb') as f:
        pickle.dump(all_procs, f)
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

    # with open('./data/med_orders.pickle', 'wb') as f:
    #     pickle.dump(data_med[['ptid', 'vid', 'Med_Pharm_Cls', 'anon_adm_date']], f)
    # f.close()
    # data_med[['ptid', 'vid', 'Med_Pharm_Cls', 'anon_adm_date']].to_csv('./data/med_orders.csv', index=False)
    #
    # with open('./data/proc_orders.pickle', 'wb') as f:
    #     pickle.dump(data_proc[['ptid', 'vid', 'PROC_ID', 'anon_adm_date']], f)
    # f.close()
    # data_proc[['ptid', 'vid', 'PROC_ID', 'anon_adm_date']].to_csv('./data/proc_orders.csv', index=False)
    # print('Done!')

    # ================================ load data back and analyze the visit in orders but not appearing in dxs ======
    # with open('./data/dxs_data.pickle', 'rb') as f:
    #     data_dx = pickle.load(f)
    # f.close()
    # dxs_visits = data_dx[['ptid', 'vid']].drop_duplicates()

    with open('./data/orders_visit_info.pickle', 'rb') as f:
        visit_info_orders = pickle.load(f)
    f.close()

    with open('./data/visits_not_in_dxs.pickle', 'rb') as f:
        vid_diff_order2dx = pickle.load(f)
    f.close()

    orders_visits = visit_info_orders[['ptid', 'vid']].drop_duplicates()

    visits_not_in_dxs = orders_visits[orders_visits['vid'].isin(vid_diff_order2dx)] # 3 rows with vid and ptid, where these vid not appeared in dxs data
    pts_with_visits_not_in_dxs = set(visits_not_in_dxs['ptid'].values) # 3 patients with visits not appeared in dxs data
    orders_pts_with_visits_not_in_dxs = orders_visits[orders_visits['ptid'].isin(pts_with_visits_not_in_dxs)]
    # 22 visits, hence all these 3 patients have no other records with dxs info in the dx data;
    # so just need to remove these 3 patients from med and proc order data

    visit_info_orders_updated = visit_info_orders[~visit_info_orders['ptid'].isin(pts_with_visits_not_in_dxs)]
    # with open('./data/orders_visit_info_updated.pickle', 'wb') as f:
    #     pickle.dump(visit_info_orders_updated, f)
    # f.close()

    # ================================ Analyze sequence lengths and how many visits before hospitalization ==========
    # with open('./data/orders_visit_info_updated.pickle', 'rb') as f:
    #     visit_info_orders_updated = pickle.load(f)
    # f.close()
    del visit_info_orders_updated['timeline']
    # remove patients with negative visit discharge
    visits = visit_info_orders_updated
    pts_negative_disch = visits[visits['anon_dis_date'] < 0]
    pts_negative_disch = set(pts_negative_disch['ptid'].values) # remove 103838 patients with negative discharge time
    visits = visits[~visits['ptid'].isin(pts_negative_disch)].drop_duplicates() # 298,479 patients
    visits = visits.sort(['ptid', 'vid', 'anon_adm_date']).drop_duplicates()
    pt_ids = list(set(visits['ptid'].values)) # 37,0074 patients

    # process the admission time and visit id:
    # OP visits with the same ids, give the earliest admission time to that visit;
    # then, visits with different ids but same times are used as same visit
    visit_adm_time = visits[['vid', 'anon_adm_date']].groupby('vid').min()
    visit_adm_time['vid'] = visit_adm_time.index
    visits_v2 = pd.merge(left=visits, right=visit_adm_time, how='inner', left_on='vid', right_on='vid')
    visit_dis_time = visits[['vid', 'anon_dis_date']].groupby('vid').max()
    visit_dis_time['vid'] = visit_dis_time.index
    visits_v2 = pd.merge(left=visits_v2, right=visit_dis_time, how='inner', left_on='vid', right_on='vid')
    del visits_v2['anon_adm_date_x']
    del visits_v2['anon_dis_date_x']
    del visits_v2['VisitDXs']
    visits_v2['cdrIPorOP'] = visits_v2['cdrIPorOP'].map({'OP': 'OP', 'IP': 'IP', 'OBS': 'OP'})
    visits_v2 = visits_v2.sort(['ptid', 'anon_adm_date_y', 'vid']).drop_duplicates() # 370074 pts
    with open('./data/visits_v2.pickle', 'wb') as f:
        pickle.dump(visits_v2, f)
    f.close()

    # # select pts with at least two visits
    # counts = visits_v2[['ptid', 'anon_adm_date_y']].drop_duplicates().groupby('ptid').count() # 370074 visits
    # counts2 = counts[counts['anon_adm_date_y'] >= 2] # 200,741 patients
    # visits_v3 = visits_v2[visits_v2['ptid'].isin(counts2.index)].drop_duplicates()
    # visits_v3 = visits_v3.sort(['ptid', 'anon_adm_date_y'])
    # # duplicated visits which each visit has more than one visit, take it as an inpatient visit
    # vids_multiple_visit_type = set(visits_v3[visits_v3.duplicated('vid')]['vid'])
    # visits_v3.ix[visits_v3['vid'].isin(vids_multiple_visit_type), 'cdrIPorOP'] = 'IP'
    # visits_v3 = visits_v3.drop_duplicates()
    # with open('./data/visits_v3.pickle', 'wb') as f:
    #     pickle.dump(visits_v3, f)
    # f.close()
    #
    # with open('./data/visits_v3.pickle', 'rb') as f:
    #     visits_v3 = pickle.load(f)
    # f.close()
    #
    # with open('./data/visits_v2.pickle', 'rb') as f:
    #     visits_v2 = pickle.load(f)
    # f.close()
    # get all patients who have a 1.5 year history
    visits_v3 = visits_v2[visits_v2['anon_dis_date_y'] >= 1.5 * 360 * 24 * 60]
    visits_v3_ids = set(visits_v3['ptid'].values) # 103363 pts
    visits_v4 = visits_v2[visits_v2['ptid'].isin(visits_v3_ids)]
    visits_v4 = visits_v4[visits_v4['anon_adm_date_y'] >= 1.5 * 360 * 24 * 60]
    with open('./data/visits_v4.pickle', 'wb') as f:
        pickle.dump(visits_v4, f)
    f.close()

    # IPvisits = visits_v4[visits_v4['cdrIPorOP'] == 'IP'] # 3677 patients with inhospital visits
    # pos_ids = list(set(IPvisits['ptid'].values)) # 10416 pts
    # neg_ids = list(visits_v3_ids.difference(set(pos_ids))) # 92947

    # ====================== Remove the unused pts from the orders and dx data =====================
    # with open('./data/dxs_data.pickle', 'rb') as f:
    #     data_dx = pickle.load(f)
    # f.close()

    # with open('./data/med_orders.pickle', 'rb') as f:
    #     data_med = pickle.load(f)
    # f.close()
    #
    # with open('./data/proc_orders.pickle', 'rb') as f:
    #     data_proc = pickle.load(f)
    # f.close()
    # ptids = pos_ids + neg_ids
    # data_dx.columns = ['ptid', 'vid', 'itemid', 'pdx']
    # data_dx = data_dx[data_dx['ptid'].isin(ptids)]
    #
    data_dx.columns = ['ptid', 'vid', 'itemid', 'pdx']
    dxgrps, dxgrps_dict, dxgrps_dict2 = dx2dxcat()
    data_dx2 = process_dxs(data_dx, dxgrps_dict, dxgrps_dict2)
    data_dx['dxcat'] = data_dx['itemid'].apply(lambda x: 'dx' + str(x)).apply(lambda x: x.replace('.', ''))
    #
    # data_med.columns = ['ptid', 'vid', 'itemid', 'adm_date']
    # data_med = data_med[data_med['ptid'].isin(ptids)]
    #
    # data_proc.columns = ['ptid', 'vid', 'itemid', 'adm_date']
    # data_proc = data_proc[data_proc['ptid'].isin(ptids)]
    # data_proc['itemid'] = data_proc['itemid'].apply(lambda x: 'p' + str(x))


    with open('./data/dxs_data_v2.pickle', 'wb') as f:
        pickle.dump(data_dx, f)
    f.close()
    data_dx.to_csv('./data/dxs_data_v2.csv', index=False)

    # with open('./data/med_orders_v2.pickle', 'wb') as f:
    #     pickle.dump(data_med[['ptid', 'vid', 'itemid']], f)
    # f.close()
    # data_med[['ptid', 'vid', 'itemid']].to_csv('./data/med_orders_v2.csv', index=False)
    #
    # with open('./data/proc_orders_v2.pickle', 'wb') as f:
    #     pickle.dump(data_proc[['ptid', 'vid', 'itemid']], f)
    # f.close()
    # data_proc[['ptid', 'vid', 'itemid']].to_csv('./data/proc_orders_v2.csv', index=False)
    #
    # with open('./data/dxs_data_v2.pickle', 'rb') as f:
    #     data_dx2 = pickle.load(f)
    # f.close()
    # with open('./data/med_orders_v2.pickle', 'rb') as f:
    #     data_med = pickle.load(f)
    # f.close()
    # with open('./data/proc_orders_v2.pickle', 'rb') as f:
    #     data_proc = pickle.load(f)
    # f.close()
    # # to merge dx, med, and proc info
    # data_dx3 = data_dx2[['ptid', 'vid', 'dxcat']].drop_duplicates()
    # data_dx3.columns = ['ptid', 'vid', 'itemid']
    # dt1 = pd.concat([data_dx3, data_med[['ptid', 'vid', 'itemid']].drop_duplicates(),
    #                  data_proc[['ptid', 'vid', 'itemid']].drop_duplicates()], axis=0)
    #
    # with open('./data/visits_v2.pickle', 'rb') as f:
    #     visits_v2 = pickle.load(f)
    # f.close()

    # dt = pd.merge(left=dt1, right=visits_v2[['vid', 'cdrIPorOP', 'anon_adm_date_y']].drop_duplicates(),
    #               left_on='vid', right_on='vid', how='inner')
    #
    # dt['adm_month'] = dt['anon_adm_date_y'].apply(lambda x: int(x/24/60/30))
    # with open('./data/clinical_events_hospitalization.pickle', 'wb') as f:
    #     pickle.dump(dt, f)
    # f.close()
    #
    # # # to exclude some inhospitalization
    # # with open('./data/clinical_events_hospitalization.pickle', 'rb') as f:
    # #     dt = pickle.load(f)
    # # f.close()
    # #
    # # with open('./data/dxs_data_v2.pickle', 'rb') as f:
    # #     dxs = pickle.load(f)
    # # f.close()
    # #
    # # dt2 = pd.merge(left=dt, right=dxs[['vid', 'pdx']],
    # #               left_on='vid', right_on='vid', how='inner')
    # # dt2_ips = dt2[dt2['cdrIPorOP'] == 'IP']
    # # dt2_ips = dt2[~dt2['pdx'].isnull()]
    # # dxgrps0, dxgrps_dict0, dxgrps_dict20 = dx2dxcat()
    # # dt2_ips2 = process_pdxs(dt2_ips, dxgrps_dict0, dxgrps_dict20)
    # # # cts = dt2_ips2[['ptid', 'pdxcat']].drop_duplicates().groupby('pdxcat').count()
    # # # cts.reset_index(inplace=True)
    # # # cts.columns = ['pdxcat', 'ct']
    # # # cts = cts.sort(['ct'], ascending=[0])
    # # # # cts2 = cts[cts['pdxcat'].isin(['49', '50', '108', '127', '158'])]
    # # # cts['pdxcat'] = cts['pdxcat'].astype(int)
    # # # cts = cts[~cts['pdxcat'].between(176, 239)]
    # # # cts = cts[~cts['pdxcat'].between(2601, 2621)]
    # # # cts['pdxcat'] = cts['pdxcat'].astype(str)
    # # dt2_ips2['pdxcat'] = dt2_ips2['pdxcat'].astype(int)
    # # dt2_ips3 = dt2_ips2[dt2_ips2['pdxcat'].between(2601, 2621)] # E codes
    # # dt2_ips4 = dt2_ips2[dt2_ips2['pdxcat'].between(212, 240)] # dxs at birth, injuries, fractures, etc.
    # # dt2_ips5 = dt2_ips2[dt2_ips2['pdxcat'] == 196]
    # # unavoid_ip_ptids = set(dt2_ips3['ptid'].values).union(set(dt2_ips4['ptid'].values)).union(set(dt2_ips5['ptid'].values))
    # # # unavoid_ip_ptids = set(dt2_ips3['ptid'].values)
    # # other_ip_ids = set(dt2_ips2['ptid'].values).difference(unavoid_ip_ptids)
    # # final_pos_ids = set(other_ip_ids).intersection(set(pos_ids)) # 4,463
    # # final_neg_ids = set(neg_ids).difference(set(unavoid_ip_ptids)) # 72,677
    # #
    # # with open('./data/hospitalization_data_pos_neg_ids.pickle', 'wb') as f:
    # #     pickle.dump([final_pos_ids, final_neg_ids], f)
    # # f.close()
    #
    # with open('./data/hospitalization_data_pos_neg_ids_v0.pickle', 'wb') as f:
    #     pickle.dump([pos_ids, neg_ids], f)
    # f.close()
    #
    # dt = dt[['ptid', 'adm_month', 'itemid', 'cdrIPorOP']].drop_duplicates()
    # dt = dt.sort(['ptid', 'adm_month'], ascending=[1, 1])
    # dt_1yr = dt[dt['adm_month'].between(0, 11)]
    #
    # with open('./data/hospitalization_data_1year.pickle', 'wb') as f:
    #     pickle.dump(dt_1yr, f)
    # f.close()
    #
    # # # get the primary dx of hospitalization of interest
    # # IPs = dt2_ips2[dt2_ips2['adm_month'] > 17]
    # # IP_adm = IPs[['ptid', 'adm_month']].drop_duplicates().groupby(['ptid']).min()
    # # IP_adm = IP_adm.reset_index()
    # # IP_adm_pdx = pd.merge(left=IPs[['ptid', 'adm_month', 'pdxcat']].drop_duplicates(), right=IP_adm,
    # #               left_on='ptid', right_on='ptid', how='inner')
    # # IP_adm_pdx = IP_adm_pdx[IP_adm_pdx['adm_month_x'] == IP_adm_pdx['adm_month_y']]
    # # del IP_adm_pdx['adm_month_y']
    # # IP_adm_pdx.columns = ['ptid', 'adm_month', 'pdxcat']
    # # with open('./data/IP_adm_month_pdx.pickle', 'wb') as f:
    # #     pickle.dump(IP_adm_pdx, f)
    # # f.close()
    #
    # # prepare data for word embedding w2v
    # with open('./data/clinical_events_hospitalization.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # f.close()
    # data = data[['vid', 'itemid']].drop_duplicates()
    # docs = data.groupby('vid')['itemid'].apply(list)
    # docs = docs.reset_index()
    # docs['length'] = docs['itemid'].apply(lambda x: len(x))
    # docs['length'].describe() # 80% quantile is 10; 92.5% quantile is 20; 98.75% quantile is 100
    # # all itemids by visit
    # vts = docs['itemid'].values.tolist()
    #
    # with open('./data/visit_items_for_w2v.pickle', 'wb') as f:
    #     pickle.dump(vts, f)
    # f.close()
    #
    # # with open('./data/hospitalization_data_pos_neg_ids.pickle', 'rb') as f:
    # #     pos_ids, neg_ids = pickle.load(f)
    # # f.close()
    # #
    # # dt_pos = dt_1yr[dt_1yr['ptid'].isin(pos_ids)]
    # # dt_neg = dt_1yr[dt_1yr['ptid'].isin(neg_ids)]
    # #
    # # counts_pos = get_counts_by_class(dt_pos, 1, 0.05 * len(pos_ids))
    # # counts_neg = get_counts_by_class(dt_neg, 0, 0.05 * len(neg_ids))
    # # counts = counts_pos.append(counts_neg).fillna(0)
    # # prelim_features = set(counts.columns[:-1])
    # #
    # # counts_sub_pos = get_counts_subwindow(dt_pos, 1, prelim_features, 3)
    # # counts_sub_neg = get_counts_subwindow(dt_neg, 0, prelim_features, 3)
    # # counts_sub = counts_sub_pos.append(counts_sub_neg).fillna(0)
    # #
    # # with open('./data/hospitalization_data_counts.pickle', 'wb') as f:
    # #     pickle.dump(counts, f)
    # # f.close()
    # #
    # # with open('./data/hospitalization_data_counts_sub.pickle', 'wb') as f:
    # #     pickle.dump(counts_sub, f)
    # # f.close()
    # # But need to work on the time window, to exclude some info before the prediction window
    # # also need to study the prediction window and how to structure it as a semi-supervised learning method
    #
    # # To do:
    # # 1. extract the dxs, meds, procedures to learn a vector representation
    # # need to think about how to structure it since it is two-level: pt level and visit level
    # # think about GloVe which uses co-occurrence counts rather than orders,
    # # There are orders between visits, and no orders within visits
    # # One solution: make each visit as a document to learn code vectors, do not consider patients
    # # The other solution is consider the visits by patients
    #
    #
    # # get the CCS categories for dx and proc codes
    # with open('./data/all_procs.pickle', 'rb') as f:
    #     all_procs = pickle.load(f)
    # f.close()
    #
    # with open('./data/all_dx_and_meds.pickle', 'rb') as f:
    #     all_dxs, all_meds = pickle.load(f)
    # f.close()
    #
    # dxgrps, dxgrps_dict, dxgrps_dict2 = dx2dxcat()
    # dx_df = process_dxs(all_dxs, dxgrps_dict, dxgrps_dict2)
    # procgrps, all_procs, proc_df = proc2proccat(all_procs)
    #
    # with open('./data/ccs_codes_dx_proc_med.pickle', 'wb') as f:
    #     pickle.dump([dx_df, proc_df, all_meds], f)
    # f.close()
    #
    # item_cats = get_all_item_categories(dx_df, proc_df, all_meds)
    # with open('./data/ccs_codes_all_item_categories.pickle', 'wb') as f:
    #     pickle.dump(item_cats, f)
    # f.close()