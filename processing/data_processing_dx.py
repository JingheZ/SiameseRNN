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
    