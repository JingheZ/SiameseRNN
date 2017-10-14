
import pandas as pd
import pickle
import seaborn as sns
import numpy as np

#
#
# def load_data(filename, colnames):
#     data = pd.read_csv(filename, delimiter='\t', header=None, encoding="ISO-8859-1")
#     data.columns = colnames
#     data['ptid'] = data['ptid'].astype(str)
#     data['vid'] = data['vid'].astype(str)
#     return data
#
# # analysis on med names
# filename = './data/2017_02feb_28_Orders_Meds.dat'
# colnames_med = ['med_order_num', 'Med_Nme', 'Med_Thera_Cls', 'Med_Pharm_Cls', 'Med_Pharm_Sub_Cls',
#                 'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date', 'anon_dis_date',
#                 'cdrIPorOP', 'order_pt_loc', 'author_physid_deid', 'order_physid_deid']
# data = load_data(filename, colnames_med)
# meds = data[['Med_Pharm_Cls', 'Med_Nme']].drop_duplicates()
# with open('./data/med_names.pickle', 'wb') as f:
#     pickle.dump(meds, f)
# print('Done!')
#
# with open('./data/med_names.pickle', 'rb') as f:
#     meds = pickle.load(f)
# meds[meds['Med_Pharm_Cls'] == 'Diagnostic Products'].to_csv('./data/med_names_diagnostic_products.csv', index=False)
#
#
# # analysis on patient ages
# with open('./data/orders_pt_info.pickle', 'rb') as f:
#     pt_info_orders = pickle.load(f)
#
# with open('./data/data_dm_ptids.pickle', 'rb') as f:
#     _, ptids_dm = pickle.load(f)
#
# ages = pt_info_orders[['ptid', 'age']].drop_duplicates().groupby('ptid').min()
# ages = ages['age'].to_dict()
# ages_pts = [ages[pid] for pid in ptids_dm]
# ages_pts = pd.Series(ages_pts, name='Age')
# ages_pts.to_csv('./data/dm_ages.csv', index=False)
#
# ages_pts = pd.read_csv('./data/dm_ages.csv', index_col=None)
# ages_pts.name = 'Age'
# ax = sns.distplot(ages_pts)

#
# with open('./results/example_pts_info.pickle', 'rb') as f:
#     exmple, exmple_seq, seq_wts_exmple, code_wts_exmple = pickle.load(f)

with open('example_pts_info.pickle', 'rb') as f:
    exmple, exmple_seq, seq_wts_exmple, code_wts_exmple = pickle.load(f)

def prepare_for_heatmap(data):
    grp = []
    val = []
    lab = []
    for i, d in enumerate(data):
        k = [d0[0] for d0 in d]
        v = [d0[1] for d0 in d]
        c = [i] * len(d)
        grp += k
        val += v
        lab += c
    df = pd.DataFrame([grp, val, lab])
    df = df.transpose()
    df.columns = ['clinical_group', 'weight', 'time']
    dt1 = df.pivot(index='clinical_group', columns='time', values='weight')
    dt1 = dt1.fillna(0)
    dt1.reset_index(inplace=True)
    return dt1
# example pt1
dt1 = prepare_for_heatmap(code_wts_exmple[0])
dt1.columns = ['name', 't1', 't2', 't4']
dt1['t3'] = 0.0
dt1 = dt1[['name', 't1', 't2', 't3', 't4']]
dt1 = dt1.sort(['t4', 't3', 't2', 't1'], ascending=[0, 0, 0, 0])
hm1 = sns.heatmap(dt1[['t1', 't2', 't3', 't4']].values, vmin=0, vmax=0.2)
hm1.savefig("example_heatmap1.png")

# example pt2
dt2 = prepare_for_heatmap(code_wts_exmple[1])
dt2.columns = ['name', 't1', 't2', 't3', 't4']
dt2 = dt2.sort(['t4', 't3', 't2', 't1'], ascending=[0, 0, 0, 0])
hm2 = sns.heatmap(dt2[['t1', 't2', 't3', 't4']].values, vmin=0, vmax=0.4)
hm2.savefig("example_heatmap2.png")

# example pt3
dt3 = prepare_for_heatmap(code_wts_exmple[2])
dt3.columns = ['name', 't1', 't3', 't4']
dt3['t2'] = 0.0
dt3 = dt3[['name', 't1', 't2', 't3', 't4']]
dt3 = dt3.sort(['t4', 't3', 't2', 't1'], ascending=[0, 0, 0, 0])
hm3 = sns.heatmap(dt3[['t1', 't2', 't3', 't4']].values, vmin=0, vmax=0.3)
hm3.savefig("example_heatmap3.png")