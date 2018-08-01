from santander_value_prediction.io import read, select_sample
import pandas as pd
import numpy as np
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

import datetime


COLUMNS = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec',
           '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12',
           '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
           '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501',
           '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
           'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
           '1931ccfdd', '703885424', '70feb1494', '491b9ee45',
           '23310aa6f', 'e176a204a', '6619d81fc', '1db387535',
           'fc99f9426', '91f701ba2', '0572565c2', '190db8488',
           'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98',
           ]


def _get_leak(df, cols, lag=0):
    """ To get leak value, we do following:
       1. Get string of all values after removing first two time steps
       2. For all rows we shift the row by two steps and again make a string
       3. Just find rows where string from 2 matches string from 1
       4. Get 1st time step of row in 3 (Currently, there is additional condition to only fetch value if we got exactly one match in step 3)"""
    series_str = df[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    series_shifted_str = df[cols].shift(lag+2, axis=1)[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    target_rows = series_shifted_str.apply(lambda x: np.where(x == series_str)[0])
    target_vals = target_rows.apply(lambda x: df.loc[df.index[x[0]], cols[lag]] if len(x)==1 else np.nan)
    return target_vals

def get_all_leak(df, cols=None, nlags=21):
    """
    We just recursively fetch target value for different lags
    """
    df = df.copy()
    for i in range(nlags):
        print("Processing lag {}".format(i))
        df["leaked_target_"+str(i)] = _get_leak(df, cols, i)
    return df


if __name__ == '__main__':
    import sys

    n_lags = int(sys.argv[1])

    df = read().set_index('ID')
    predictors = [c for c in df.columns if c not in ['isTrain', 'target']]
    df_train = select_sample(df, "train")
    df_test = select_sample(df, "test")

    df_leaked = get_all_leak(df[predictors], COLUMNS, n_lags)
    cols_leaked = [c for c in df_leaked.columns if c.startswith('leaked_target')]

    with open('train/df_leaked.pyt', 'wb') as pyt:
        joblib.dump(df_leaked[cols_leaked], pyt)

    ugly_lines = joblib.load('data/ugly_lines_test.pyt')
    ugly_rows = df_test.index[ugly_lines]
    private = np.where(df_leaked.loc[ugly_rows, cols_leaked].replace(np.nan, 0).sum(axis=1) == 0)[0]
    private_lines = ugly_lines[private]

    with open('train/private_lines_lag-%i.pyt' % n_lags, 'wb') as pyt:
        joblib.dump(private_lines, pyt)



