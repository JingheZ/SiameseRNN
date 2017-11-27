
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename, colnames):
    data = pd.read_csv(filename, delimiter='\t', header=None, encoding="ISO-8859-1")
    data.columns = colnames
    data['ptid'] = data['ptid'].astype(str)
    data['vid'] = data['vid'].astype(str)
    return data

# analysis on med names
filename = './data/2017_02feb_28_Orders_Meds.dat'
colnames_med = ['med_order_num', 'Med_Nme', 'Med_Thera_Cls', 'Med_Pharm_Cls', 'Med_Pharm_Sub_Cls',
                'ptid', 'age', 'sex', 'vid', 'VisitDXs', 'timeline', 'anon_adm_date', 'anon_dis_date',
                'cdrIPorOP', 'order_pt_loc', 'author_physid_deid', 'order_physid_deid']
data = load_data(filename, colnames_med)
meds = data[['Med_Pharm_Cls', 'Med_Nme']].drop_duplicates()
with open('./data/med_names.pickle', 'wb') as f:
    pickle.dump(meds, f)
print('Done!')

with open('./data/med_names.pickle', 'rb') as f:
    meds = pickle.load(f)
meds[meds['Med_Pharm_Cls'] == 'Diagnostic Products'].to_csv('./data/med_names_diagnostic_products.csv', index=False)


# analysis on patient ages
with open('./data/orders_pt_info.pickle', 'rb') as f:
    pt_info_orders = pickle.load(f)

with open('./data/data_dm_ptids.pickle', 'rb') as f:
    _, ptids_dm = pickle.load(f)

ages = pt_info_orders[['ptid', 'age']].drop_duplicates().groupby('ptid').min()
ages = ages['age'].to_dict()
ages_pts = [ages[pid] for pid in ptids_dm]
ages_pts = pd.Series(ages_pts, name='Age')
ages_pts.to_csv('./data/dm_ages.csv', index=False)

ages_pts = pd.read_csv('./data/dm_ages.csv', index_col=None)
ages_pts.name = 'Age'
ax = sns.distplot(ages_pts)

#
# with open('./results/example_pts_info.pickle', 'rb') as f:
#     exmple, exmple_seq, seq_wts_exmple, code_wts_exmple = pickle.load(f)
#
# with open('./results/example_pts_info_v2.pickle', 'wb') as f:
#     pickle.dump([seq_wts_exmple, code_wts_exmple], f )

with open('example_pts_info_v2.pickle', 'rb') as f:
    seq_wts_exmple, code_wts_exmple = pickle.load(f)

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
dt1 = dt1.sort_values(['name', 't4', 't3', 't2', 't1'], ascending=[1, 0, 0, 0, 0])
dt1.index = dt1['name']
del dt1['name']
dt1.loc['Weight'] = seq_wts_exmple[0]
dt1 = dt1.ix[~dt1['t4'].between(0.01, 0.017)]
dt1.index = list(dt1.index[:10]) + ['Coronary atherosclerosis & heart disease', 'Conduction disorders',
                                    'Cardiac dysrhythmias', 'Diverticulosis and diverticulitis',
                                    'Gastrointestinal hemorrhage', 'Acute and unspecified renal failure',
                                    'Chronic kidney disease', 'Hyperplasia of prostate', 'Osteoarthritis',
                                    'Other connective tissue disease', 'Other aftercare', 'Nutritional deficiencies',
                                    'Disorders of lipid metabolism', 'Nutritional, endocrine, metabolic disorders',
                                    'Deficiency and other anemia',
                                    'Screening of mental health & substance abuse',
                                    'Dizziness or vertigo',
                                    'Hypertension with complications'] + \
            ['Suture of skin and subcutaneous tissue', 'Other organ transplantation', 'Arterial blood gases',
             'Microscopic examination', 'Diagnostic radiology and related techniques',
             'Laboratory - Chemistry and hematology', 'Pathology', 'Other laboratory',
             'OR therapeutic procedures; nose, mouth, pharynx', 'Other OR heart procedures',
             'Embolectomy and endarterectomy of lower limbs', 'Therapeutic procedures; hemic & lymphatic system',
             'OR therapeutic nervous system procedures', 'sequence-level weight']
dt1.to_csv('./results/heatmap1_data.csv')
plt.figure(figsize=(32, 15))
sns.set(font_scale=1.2)
hm1 = sns.heatmap(dt1, vmin=0, vmax=0.35)
plt.yticks(rotation=0)
plt.xlabel('Time')
plt.ylabel('')
plt.savefig("./results/example_heatmap1.png")

# example pt2
dt2 = prepare_for_heatmap(code_wts_exmple[1])
dt2.columns = ['name', 't1', 't2', 't3', 't4']
dt2 = dt2.sort_values(['name', 't4', 't3', 't2', 't1'], ascending=[1, 0, 0, 0, 0])
dt2.index = dt2['name']
del dt2['name']
dt2.loc['Weight'] = seq_wts_exmple[1]
dt2.index = ['Diagnostic Products', 'Coronary atherosclerosis & heart disease',
             'Genitourinary symptoms', 'Spondylosis', 'Other connective tissue disease',
             'Bone disease & musculoskeletal deformities', 'Malaise and fatigue',
             'Diabetes mellitus without complication', 'Diabetes mellitus with complications',
             'Disorders of lipid metabolism', 'Essential hypertension',
             'Diagnostic procedures', 'Other organ transplantation',
             'Therapeutic procedures on conjunctiva; cornea', 'Arterial blood gases',
             'Microscopic examination', 'Other radioisotope scan',
             'Other laboratory', 'Other OR heart procedures',
             'Non-OR therapeutic nervous system procedures', 'Sequence-level weight']
dt2.to_csv('./results/heatmap2_data.csv')
plt.figure(figsize=(30, 15))
sns.set(font_scale=1.2)
hm2 = sns.heatmap(dt2[['t1', 't2', 't3', 't4']], vmin=0, vmax=0.35)
plt.yticks(rotation=0)
plt.xlabel('Time')
plt.ylabel('')
plt.savefig("./results/example_heatmap2.png")

# example pt3
dt3 = prepare_for_heatmap(code_wts_exmple[2])
dt3.columns = ['name', 't1', 't3', 't4']
dt3['t2'] = 0.0
dt3 = dt3[['name', 't1', 't2', 't3', 't4']]
dt3 = dt3.sort(['name', 't4', 't3', 't2', 't1'], ascending=[1, 0, 0, 0, 0])
dt3.index = dt3['name']
del dt3['name']
dt3.loc['Weight'] = seq_wts_exmple[2]
plt.figure(figsize=(15, 16))
hm3 = sns.heatmap(dt3[['t1', 't2', 't3', 't4']], vmin=0, vmax=0.35)
plt.yticks(rotation=0)
plt.xlabel('Time')
plt.ylabel('')
hm3.savefig("./results/example_heatmap3.png")


# ====== number of visits in the observation window ========
with open('./data/clinical_events_hospitalization.pickle', 'rb') as f:
    dt = pickle.load(f)
dt = dt[dt['adm_month'].between(0, 11)]
dtvt_cts = dt[['ptid', 'vid']].drop_duplicates().groupby('ptid').count()
dtvt_cts.reset_index(inplace=True)
dtvt_cts.to_csv('./data/hospitalization_data_visit_counts.csv', index=False)

dtvt_cts = pd.read_csv('./data/hospitalization_data_visit_counts.csv', index_col=None)
dtvt_cts['# visits'] = dtvt_cts['vid']
ax = sns.distplot(dtvt_cts['# visits'], hist_kws=dict(cumulative=True),
             kde_kws=dict(cumulative=True))
ax.set(xlim=(0, 50), xticks=np.arange(0,51,5))



