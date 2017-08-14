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
    return dxgrps


def process_dxs(data, dxgrps):
    data['dx'] = data['itemid'].str.replace('.', '')
    dxs = data['dx'].values.tolist()
    dxcats = []
    for i in dxs:
        if i in dxgrps.index:
            dxcats.append(dxgrps['category'].loc[i])
        else:
            dxcats.append('0')
    data['dxcat'] = dxcats
    return data


def find_first_visit(data_dx, visits):
    data = pd.merge(data_dx, visits, how='inner', left_on='visitid', right_on='visitid')
    data.sort(['ptid', 'adm_date'], ascending=[1, 1], inplace=True)
    first_visit = data[['ptid', 'adm_date']].drop_duplicates().groupby('ptid').min()
    first_visit.columns = ['ptid', 'first_visit_date']
    data_v2 = pd.merge(data, first_visit, how='inner', left_on='ptid', right_on='ptid')
    data_v2['gap_1st'] = data_v2['adm_date'] - data_v2['first_visit_date']
    return data_v2


def find_visit_gaps(data, dxcats):
    dms = data[data['dxcat'].isin(dxcats)]
    len(set(dms['ptid']))
    first_dm = dms[['ptid', 'adm_date']].drop_duplicates().groupby('ptid').min()
    first_dm.columns = ['ptid', 'first_dm_date']
    data_v2 = pd.merge(data, first_dm, how='inner', left_on='ptid', right_on='ptid')
    data_v2['gap_dm'] = data_v2['first_dm_date'] - data_v2['adm_date']
    data_v2['gap_1st_to_dm'] = data_v2['first_dm_date'] - data_v2['first_visit_date']
    return data_v2


def find_patient_counts(data):
    x1 = data[data['gap_1st_to_dm'] >= 90]
    print('Number of patients with first DM after 90 days % i:')
    print(len(set(x1['ptid'])))
    x2 = data[data['gap_1st_to_dm'] >= 180]
    print('Number of patients with first DM after 180 days % i:')
    print(len(set(x2['ptid'])))
    x3 = data[data['gap_1st_to_dm'].between(90, 455)]
    print('Number of patients with first DM between 90 and 455 days % i:')
    print(len(set(x3['ptid'])))
    x4 = data[data['gap_1st_to_dm'].between(180, 635)]
    print('Number of patients with first DM between 180 and 455 days % i:')
    print(len(set(x4['ptid'])))


if __name__ == '__main__':
    # ============================ DX Data =================================================
    with open('./data/visits_v4.pickle', 'rb') as f:
        visits = pickle.load(f)
    f.close()

    with open('./data/dxs_data_v2.pickle', 'rb') as f:
        data_dx = pickle.load(f)
    f.close()

    dxgrps = dx2dxcat()
    data_dx2 = process_dxs(data_dx, dxgrps)

    data = find_first_visit(data_dx2, visits)
    # find patients with diabetes: dxcat = '49' or '50'
    data_dm = find_visit_gaps(data, ['49', '50'])
    find_patient_counts(data_dm)

    # find patients with CHF: dxcat = '108'
    data_dm = find_visit_gaps(data, ['108'])
    find_patient_counts(data_dm)

