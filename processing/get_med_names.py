
import pandas as pd
import pickle
import seaborn as sns


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
meds


# analysis on patient ages
with open('./data/orders_pt_info.pickle', 'rb') as f:
    pt_info_orders = pickle.load(f)

with open('./data/data_dm_ptids.pickle', 'rb') as f:
    _, ptids_dm = pickle.load(f)

ages = pt_info_orders[['ptid', 'age']].drop_duplicates().groupby('ptid').min()
ages = ages['age'].to_dict()
# ptids = list(pos_ids) + list(neg_ids)
# ptids = train_ids
ages_pts = [ages[pid] for pid in ptids_dm]
ages_pts = pd.Series(ages_pts, name='Age')
ages_pts.to_csv('./data/dm_ages.csv', index=False)
ax = sns.distplot(ages_pts)

.quantile(0.3)
