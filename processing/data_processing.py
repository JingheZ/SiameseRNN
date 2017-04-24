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
    visit_info_orders = pd.concat((visit_info_med, visit_info_proc), axis=1)
    visit_info_orders = visit_info_orders.drop_duplicates()
    with open('./data/orders_visit_info.pickle', 'wb') as f:
        pickle.dump(visit_info_orders, f)
    f.close()
    # return visit_info_orders


def get_orders_pt_info(data_med, data_proc):
    pt_info_med = data_med[['ptid', 'age', 'sex']]
    pt_info_proc = data_proc[['ptid', 'age', 'sex']]
    pt_info_orders = pd.concat((pt_info_med, pt_info_proc), axis=1)
    pt_info_orders = pt_info_orders.drop_duplicates()
    with open('./data/orders_pt_info.pickle', 'wb') as f:
        pickle.dump(pt_info_orders, f)
    f.close()
    # return pt_info_orders


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

    # ============================== Save data to pickle ========================================================
    with open('./data/dxs_data.pickle', 'wb') as f:
        pickle.dump(data_dx, f)
    f.close()

    with open('./data/med_orders.pickle', 'wb') as f:
        pickle.dump(data_med[['ptid', 'vid', 'Med_Pharm_Cls']], f)
    f.close()

    with open('./data/proc_orders.pickle', 'wb') as f:
        pickle.dump(data_proc[['ptid', 'vid', 'PROC_ID']], f)
    f.close()
    
    print('Done!')
