"""
Define a cutoff, observation window (one-year) and prediction window (one-year)
"""

import pickle
import pandas as pd


if __name__ == '__main__':

    with open('./data/visits_v4.pickle', 'rb') as f:
        visits = pickle.load(f)
    f.close()

    visits = visits[['ptid', 'rank', 'anon_adm_date_y']]
    visits['adm_day'] = visits['anon_adm_date_y'] / 60 / 24
    max_adm = visits[['ptid', 'adm_day']].groupby('ptid').max()
    max_adm.describe()
