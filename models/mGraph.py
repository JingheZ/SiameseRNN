from fileinput import filename

__author__ = 'jinghe'

'''
select the patients data according to the patient ids from both med and procedure datasets
'''

"""import packages"""
import pickle
import pandas as pd
import numpy as np


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


def process_dxs(data, dxgrps_dict, dxgrps_dict2):
    data['dx'] = data['itemid'].str.replace('.', '')
    dxs = data['dx'].values.tolist()
    dxcats = []
    for i in dxs:
        if dxgrps_dict.__contains__(i):
            dxcats.append(dxgrps_dict[i])
        elif dxgrps_dict2.__contains__(i):
            dxcats.append(dxgrps_dict2[i])
        else:
            dxcats.append('0')
    data['dxcat'] = dxcats
    data = data[data['dxcat'] != '0']
    return data


def merge_visit_and_dx(data_dx, visits):
    data = pd.merge(data_dx, visits, how='inner', left_on='vid', right_on='vid')
    data = data[['ptid_x', 'vid', 'pdx', 'dxcat', 'adm_date', 'dis_date', 'rank']]
    data.columns = ['ptid', 'vid', 'pdx', 'dxcat', 'adm_date', 'dis_date', 'rank']
    data.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    return data


def find_visit_gaps(data, dxcats):
    dms = data[data['dxcat'].isin(dxcats)]
    dm_ptids = set(dms['ptid'])
    print('%i patients' % len(set(dms['ptid'])))
    first_dm = dms[['ptid', 'adm_date']].drop_duplicates().groupby('ptid').min()
    first_dm.reset_index(inplace=True)
    first_dm.columns = ['ptid', 'first_dm_date']
    data_v2 = pd.merge(data, first_dm, how='inner', left_on='ptid', right_on='ptid')
    data_v2['gap_dm'] = data_v2['first_dm_date'] - data_v2['adm_date']
    return data_v2, dm_ptids


def find_patient_counts(data):
    x1 = data[data['first_dm_date'] >= 90 * 24 * 60]
    print('Number of patients with first DM after 90 days % i:')
    print(len(set(x1['ptid'])))
    x2 = data[data['first_dm_date'] >= 180 * 24 * 60]
    print('Number of patients with first DM after 180 days % i:')
    print(len(set(x2['ptid'])))
    x3 = data[data['first_dm_date'].between(90 * 24 * 60, 455 * 24 * 60)]
    print('Number of patients with first DM between 90 and 455 days % i:')
    print(len(set(x3['ptid'])))
    x4 = data[data['first_dm_date'].between(180 * 24 * 60, 635 * 24 * 60)]
    print('Number of patients with first DM between 180 and 455 days % i:')
    print(len(set(x4['ptid'])))


def find_visit_gaps_control(data, target_ids, thres):
    # select patients with the records of the first visit
    pts_first_visit = data[data['adm_date'] == 0]
    ptids = set(pts_first_visit['ptid'])
    data = data[data['ptid'].isin(ptids)]
    # select patients visits of at least of threshold lengths of time
    pts_later_visits = data[data['adm_date'] > thres]
    ptids = set(pts_later_visits['ptid'])
    data = data[data['ptid'].isin(ptids)]
    # remove patients with the target dx
    data = data[~data['ptid'].isin(target_ids)]
    print('%i patients' % len(set(data['ptid'])))
    return data


if __name__ == '__main__':
    # ============================ DX Data =================================================
    with open('./data/visits_v4.pickle', 'rb') as f:
        visits = pickle.load(f)
    f.close()
    visits.columns = ['ptid', 'vid', 'IPorOP', 'adm_date', 'dis_date', 'rank']
    with open('./data/dxs_data_v2.pickle', 'rb') as f:
        data_dx = pickle.load(f)
    f.close()

    dxgrps, dxgrps_dict, dxgrps_dict2 = dx2dxcat()
    data_dx2 = process_dxs(data_dx, dxgrps_dict, dxgrps_dict2)
    data_dx2.head()

    data = merge_visit_and_dx(data_dx2, visits)
    # find patients with diabetes: dxcat = '49' or '50'
    data_dm, ptids_dm = find_visit_gaps(data, ['49', '50'])
    find_patient_counts(data_dm)

    # # find patients with CHF: dxcat = '108'
    # data_chf = find_visit_gaps(data, ['108'])
    # find_patient_counts(data_chf)
    #
    # # find patients with CKD: dxcat = '158'
    # data_ckd = find_visit_gaps(data, ['158'])
    # find_patient_counts(data_ckd)
    #
    # # find patients with CKD: dxcat = '127'
    # data_copd = find_visit_gaps(data, ['127'])
    # find_patient_counts(data_copd)

    # find patients with at least four years of complete visits
    # 1. first visit date = 0
    # 2. one year of observation window and four years of prediction window
    thres = 60 * 24 * 365 * 5
    data_control = find_visit_gaps_control(data, ptids_dm, thres)

    # select the visits of target and control groups: