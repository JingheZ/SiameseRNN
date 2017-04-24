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



if __name__ == '__main__':
    filename = './data/2016_04apr_13_ALL_Diagnoses.dat'
    data_dx = load_data(filename, colnames=['ptid', 'vid', 'dx', 'primary_dx'])
    visits_dx = count_stat_summary(data_dx)

    filename = './data/2016_04apr_13_Orders_Meds_redo.dat'
    colnames_med = ['med_order_num', 'Med_Nme', 'Med_Thera_Cls', 'Med_Pharm_Cls', 'Med_Pharm_Sub_Cls',
                    'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'anon_adm_date', 'anon_dis_date', 'timeline',
                    'cdrIPorOP', 'order_pt_loc', 'author_physid_deid', 'order_physid_deid']
    data_med = load_data(filename, colnames=colnames_med)
    data_med = data_med[['Med_Pharm_Cls', 'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date',
                         'anon_dis_date', 'cdrIPorOP']]
    visits_med = data_med['vid'].unique()

    filename = './data/2016_04apr_13_Orders_Procedures_redo.dat'
    colnames_proc = ['proc_order_num', 'PROC_ID', 'PROC_NAME', 'PROC_CAT', 'PROC_CAT_ID', 'ORDER_DISPLAY_NAME',
                    'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date', 'anon_dis_date',
                    'cdrIPorOP', 'physid_deid']
    data_proc= load_data(filename, colnames=colnames_proc)
    data_proc = data_proc[['proc_order_num', 'PROC_ID', 'PROC_NAME', 'PROC_CAT', 'PROC_CAT_ID', 'ORDER_DISPLAY_NAME',
                    'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date', 'anon_dis_date',
                    'cdrIPorOP', 'physid_deid']]
    visits_proc = data_proc['vid'].unique()

    vid_orders = set(visits_proc.tolist()).union(set(visits_med.tolist()))
    vid_diff_order2dx = vid_orders.difference(set(visits_dx.tolist()))
    vid_diff_dx2order = set(visits_dx.tolist()).difference(vid_orders)
