#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, Binarizer, QuantileTransformer, PolynomialFeatures, OneHotEncoder


class SortedLabelEncoder:

    '''Like a LabelEncoder but which sorts categories
    depending on the predictand mean value
    '''

    def __init__(self, predictand):
        self.unique_sorted = None
        self._predictand = predictand

    def fit(self, df, column):
        target = self._predictand
        unique = df[column].unique()
        means = df.groupby(column)[target].mean()
        self.unique_sorted = list(means.sort_values().index)

    def transform(self, series):
        return series.apply(lambda x:
                                self.unique_sorted.index(x)
                                if x in self.unique_sorted else -1)


def encode_dataframe(df, categorical_features, encoders=None):
    '''blah'''
    encoders = {}
    for col in df.columns:
        # Change dtype
        if df[col].dtype.str == "|O":
            print('Converting column \033[91m%s\033[0m of type %s to str' % (col, df[col].dtype))
            df[col] = df[col].astype(str)

        # Categorical features are randomly encoded for visualization
        if col in categorical_features:
            lb = encoders.get(col, None)
            if lb is None:
                lb = LabelEncoder()
                df[col] = lb.fit_transform(df[col])
                encoders[col] = lb
            else:
                df[col] = lb.transform(df[col])

    return df, encoders


def fill_na(df, **kwargs):
    import pdb

    for col in df.columns:
        cfg = kwargs.get(col, None)

        if isinstance(cfg, dict):
            operation = cfg.get('operation', 'median')
            by_similar = cfg.get('by_similar', [])

            if operation == "median":
                value_all = df[col].median()
            elif operation == "top":
                value_all = df[col].describe()["top"]
            else:
                raise NameError("Operation %s unknown" % operation)

            inds = np.where(pd.isnull(df[col]))[0]

            for i in inds:
                jinds = np.ones_like(df.index)
                for c in by_similar:
                    jinds &= df[c] == df[c].iloc[i]

                jinds = np.where(jinds)[0]

                if operation == "median":
                    value_similar = df[col].iloc[jinds].median()
                elif operation == "top":
                    desc = df[col].iloc[jinds].describe()
                    value_similar = desc.get('top', np.nan)

                df[col].iloc[i] = value_all if pd.isnull(value_similar) else value_similar

        elif cfg == "median":
            df[col] = df[col].fillna(df[col].median())

        elif cfg == None:
            continue
        else:
            raise NameError("Fill Na method %s not recognized" % cfg)

    return df


def feature_engineering(df, **kwargs):
    '''
    Categorical features are OneHot-encoded

    Numerical features are (if specified as such):
        - "quantilized" (cf QuantileTransformer) first,
        - then, "robust_scaled",
        - then, standardized
        - then, polynomial features are computed,
        - finally, we can either take the log(X) or log(-X)
            for right / left-skewed features resp.
    '''
    verbose = kwargs.get('verbose', False)

    right_skewed = kwargs.get('right_skewed', [])
    left_skewed = kwargs.get('left_skewed', [])

    robust_scale = kwargs.get('robust_scale', [])
    standardize = kwargs.get('standardize', [])
    quantilize = kwargs.get('quantilize', [])

    polynomials = kwargs.get('polynomials', [])

    id_features = kwargs.get('id_features', [])
    categorical_features = kwargs.get('categorical_features', [])
    ordinal_features = kwargs.get('ordinal_features', [])

    features_to_remove = kwargs.get('remove', [])

    transformers = kwargs.get('transformers', defaultdict(dict))
    for col in df.columns:

        if col in id_features:
            continue

        if col in ordinal_features:
            continue

        if col in categorical_features:
            # Use OneHotEncoder
            transf = transformers[col].get('OneHot', None)
            if transf is None:
                transf = OneHotEncoder()
                transf.fit(np.expand_dims(df[col].values, axis=1))
            else:
                if verbose:
                    print('Category %s: fetched existing OneHotEncoder' % col)

            res = transf.transform(np.expand_dims(df[col].values, axis=1))
            for i, ax in enumerate(res.transpose(), 1):
                df['{}_{}'.format(col, i)] = ax.toarray().squeeze()
            features_to_remove.append(col)
            transformers[col]['OneHot'] = transf

        if col in quantilize:
            if verbose: print('[[QuantileTransformer on %s]]' % col)
            kw = kwargs.get('{}_quantiletransformer_options'.format(col), {})

            transf = transformers[col].get('QuantileTransformer', None)
            if transf is None:
                transf = QuantileTransformer(**kw)
                transf.fit(np.expand_dims(df[col], axis=1))
            else:
                if verbose:
                    print('\tFetched existing transformer')

            df[col] = transf.transform(
                np.expand_dims(df[col], axis=1))
            transformers[col]['QuantileTransformer'] = transf

        if col in robust_scale:
            if verbose: print('[[RobustScaler on %s]]' % col)
            kw = kwargs.get('{}_robustscaler_options'.format(col), {})

            transf = transformers[col].get('RobustScaler', None)
            if transf is None:
                transf = RobustScaler(**kw)
                transf.fit(np.expand_dims(df[col], axis=1))
            else:
                if verbose:
                    print('\tFetched existing transformer')

            df[col] = transf.transform(
                np.expand_dims(df[col], axis=1))
            transformers[col]['RobustScaler'] = transf

        if col in standardize:
            if verbose: print('[[StandardScaler on %s]]' % col)
            kw = kwargs.get('{}_standardscaler_options'.format(col), {})

            transf = transformers[col].get('StandardScaler', None)
            if transf is None:
                transf = StandardScaler(**kw)
                transf.fit(np.expand_dims(df[col], axis=1))
            else:
                if verbose:
                    print('\tFetched existing transformer')

            df[col] = transf.transform(
                np.expand_dims(df[col], axis=1))
            transformers[col]['StandardScaler'] = transf

    for polynomial in polynomials:
        cols = polynomial['columns']
        degree = polynomial['degree']

        pf = PolynomialFeatures(degree)
        res = pf.fit_transform(df[cols])
        for i, ax in enumerate(res.transpose(), 1):
            ax = QuantileTransformer().fit_transform(
                np.expand_dims(ax, axis=1))
            df['_'.join(cols + [str(degree), str(i)])] = ax

    for col in df.columns:

        if col in right_skewed:
            df[col] = np.log(df[col] - df[col].min() + 1)

        elif col in left_skewed:
            df[col] = np.log(df[col].max() - df[col] + 1)


    for col in features_to_remove:
        del df[col]


    return df, transformers
