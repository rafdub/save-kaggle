#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import binarize

from santander_value_prediction.setup import DATA_PATH, FILES, FILE_OPTIONS


def read(kind="both"):
    '''useful to load both samples, to setup the encoders'''
    if kind in ['train', 'test']:
        fname = os.path.join(DATA_PATH, FILES.get(kind, kind))
        df = pd.read_csv(fname, **FILE_OPTIONS)
    else:
        df_train = read("train")
        df_test = read("test")
        df_train['isTrain'] = True
        df_test['isTrain'] = False
        df_train.index += 1
        df_test.index += df_train.index.max() + 1
        df = df_train.append(df_test)

    #df['BsmtCond'] = df['BsmtCond'].fillna('NoBsmt') # we do that here for raw analysis
    #df['BsmtExposure'] = df['BsmtExposure'].fillna('NoBsmt') # we do that here for raw analysis
    return df


def select_sample(df, kind="train"):
    '''kind = train or test'''
    boo = kind == "train"
    return df.groupby('isTrain').get_group(boo).drop('isTrain', axis=1)
