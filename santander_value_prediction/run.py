#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from itertools import combinations

from santander_value_prediction.io import read, select_sample
from santander_value_prediction.setup import DATA_PATH, ID, PREDICTAND, ANALYSIS_PATH, CATEGORICAL_FEATURES, ORDINAL_FEATURES, TRAIN_PATH, IGNORE_FEATURES, FILL_NA_CONFIG, OUTPUT_PATH, CFG_BIVARIATE_RAW, CFG_BIVARIATE_PP, NPC, CORR_THRESHOLD
from santander_value_prediction.transform import tsne, plot_pca

from commons.transform import encode_dataframe, fill_na, feature_engineering
from commons.fit import train_and_validate

from sklearn.preprocessing import binarize, LabelEncoder, QuantileTransformer, RobustScaler, OneHotEncoder

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut

from sklearn.cluster import DBSCAN

from sklearn.metrics import make_scorer, mean_squared_error

from scipy.stats import ks_2samp
from scipy.cluster import hierarchy
import scipy.spatial as ss

from traceback import print_exc


N_LAGS = 25

COLUMNS_LEAK = [
    'f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec',
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



def analyze(df, predictors, step='raw'):

    dname = os.path.join(ANALYSIS_PATH, step)
    os.makedirs(dname, exist_ok=True)
    for col in predictors:
        print('\033[1m%s: %i missing values\033[0m [of %i]' % (col, pd.isnull(df[col]).sum(), len(df)))
        #if col not in CATEGORICAL_FEATURES:
            #print('\tIgnoring column \033[91m%s\033[0m which is not categorical [%i unique values]' % (
                #col, len(df[col].unique())))
            #continue

        print('\tMaking figure for %s...' % col)
        fname = os.path.join(dname, "%s.png" % col)

        if col in CATEGORICAL_FEATURES:
            g = sns.catplot(x=col, y=PREDICTAND, data=df, kind="box")
        else:
            g = sns.jointplot(col, PREDICTAND, data=df, kind="scatter")

        corr = np.corrcoef(df[col].values, df[PREDICTAND].values)[0, 1]

        plt.xticks(rotation=90)
        plt.title('%s, %i missing values, correlation %.3f' % (col, pd.isnull(df[col]).sum(), corr))
        plt.tight_layout()
        #plt.show()
        g.savefig(fname)
        print('\tSaved figure \033[92m%s\033[0m' % fname)
        plt.close()


def analyze_bivariate(df, cfgs, step='raw'):

    dname = os.path.join(ANALYSIS_PATH, step, "bivariate")
    os.makedirs(dname, exist_ok=True)
    for cfg in cfgs:
        x_name = cfg.pop('x')
        y_name = cfg.pop('y')
        c_name = cfg.pop('c', PREDICTAND)

        print('\tMaking figure for %s as a function of %s and %s...' % (c_name, x_name, y_name))
        fname = os.path.join(dname, "%s_%s-%s.png" % (c_name, x_name, y_name))


        sub = df[[x_name, y_name, c_name]].copy()
        if step == "raw":
            sub[x_name] = sub[x_name].fillna('Na')
            sub[y_name] = sub[y_name].fillna('Na')
        sub = sub.dropna()

        values = sub.groupby((x_name, y_name))[c_name].mean().unstack()
        counts = np.log(sub.groupby((x_name, y_name))[c_name].count().unstack() + 1)

        cfg.setdefault('square', True)
        cfg.setdefault('linewidths', .5)
        cfg.setdefault('cbar_kws', {'shrink': .5})
        cfg.setdefault('cmap', cm.plasma)

        fig, axs = plt.subplots(figsize=(8, 12), nrows=2)
        g = sns.heatmap(values, ax=axs[0], **cfg)
        plt.sca(axs[0])
        plt.xticks(rotation=90)
        plt.title(c_name)

        cfg['cmap'] = cm.plasma
        g = sns.heatmap(counts, ax=axs[1], **cfg)
        plt.sca(axs[1])
        plt.xticks(rotation=90)
        plt.title("Count")

        plt.tight_layout()

        plt.savefig(fname)
        print('\tSaved figure \033[92m%s\033[0m' % fname)
        plt.close()


def multibinarize(x, thresholds):
    if hasattr(x, "fillna"):
        x = x.fillna(0).values.reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)
    res = None
    for threshold in thresholds:
        if res is None:
            res = binarize(x, threshold)
        else:
            res += binarize(x, threshold)
    return res[:, 0]


def _get_height_at(branch, n_clusters):
    if n_clusters <= 1 or branch.is_leaf():
        # Stop if we asked for the height for 1 cluster
        # or if we reached a leaf.
        return branch.dist
    else:
        nodes = [branch.left, branch.right]

        while len(nodes) < n_clusters:
            i = np.argmax([n.dist for n in nodes])
            node_to_split = nodes.pop(i)
            if node_to_split.is_leaf():
                break
            nodes.extend([node_to_split.left, node_to_split.right])

        return np.max([n.dist for n in nodes])


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


def main(verbose=True, force=False, test=False):
    import datetime

    IGNORE_FEATURES = []

    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    os.makedirs(TRAIN_PATH, exist_ok=True)

    raw_df_name = os.path.join(TRAIN_PATH, 'data_raw.pyt')
    scaled_df_name = os.path.join(TRAIN_PATH, 'data_scaled.pyt')

    st_time = datetime.datetime.now()
    print('Loading the data...')
    if not os.path.isfile(raw_df_name) or force:
        df = read()
        df.set_index(ID, inplace=True)
        print('\tWriting \033[92m%s\033[0m' % (raw_df_name))
        with open(raw_df_name, 'wb') as pyt:
            joblib.dump(df, pyt)
    else:
        print('\tLoading data from \033[92m%s\033[0m' % (raw_df_name))
        df = joblib.load(raw_df_name)
    print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())

    # log-scale the predictors & predictand
    bins_target = np.logspace(
        np.log10(df[PREDICTAND].min()),
        np.log10(df[PREDICTAND].max()),
        20)
    df[PREDICTAND] = np.log1p(df[PREDICTAND])
    predictors = [c for c in df.columns if c not in ['isTrain', PREDICTAND]]

    # Counts of 0s or non-0s is very different between test and train sets !
    pstep = 5
    percs = np.arange(pstep, 100, pstep)

    calculated_cols = []
    columns_then = df.columns


    # Add the info relative to the leak as it affects the training / test processes
    leak_file = os.path.join(TRAIN_PATH, "df_leaked_%s.pyt" % N_LAGS)
    if os.path.isfile(leak_file):
        df_leaked = joblib.load(leak_file)
    else:
        df_ = df[predictors].reset_index(level=0)
        df_[PREDICTAND] = df[PREDICTAND]
        df_ = df_[['ID', PREDICTAND] + predictors]

        df_leaked = get_all_leak(df_, COLUMNS_LEAK, N_LAGS)
        leak_cols = [c for c in df_leaked if c.startswith('leak')]
        df_leaked = df_leaked[leak_cols]
        with open(leak_file, 'wb') as pyt:
            joblib.dump(df_leaked, pyt)

    df_leaked.index = df.index
    leak_cols = df_leaked.columns
    df['nb_potential_leaks'] = df_leaked.notnull().sum(axis=1)
    df['leak_mean'] = df_leaked.mean(axis=1).fillna(0)
    df['leak_median'] = df_leaked.median(axis=1).fillna(0)
    df['leak_max'] = df_leaked.max(axis=1).fillna(0)
    df['leak_min'] = df_leaked.min(axis=1).fillna(0)


    # Clustering on sorted dataframe (row by row) to detect similar entries
    df_ = df[predictors].copy()
    for row in range(len(df_)):
        arr = df_.iloc[row, :]
        df_.iloc[row, :] = np.sort(arr)

    # Hierarchical clustering seems to have a predictive power
    #distance = "euclidean"
    n_clusters = 12
    for distance in ["hamming", "jaccard", "sokalmichener", "sokalsneath", "euclidean"]:
        st_time = datetime.datetime.now()
        print('Finding \033[92m%i clusters\033[0m with \033[92m%s distance\033[0m' % (n_clusters, distance))
        dist_fname = os.path.join(TRAIN_PATH, "%s_dists.pyt" % distance)
        if os.path.isfile(dist_fname):
            dist = joblib.load(dist_fname)
            print('-- Pairwise distance loading took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())
        else:
            if distance == "euclidean":
                dist = ss.distance.pdist(df_[predictors].values, distance)
            else:
                dist = ss.distance.pdist(df[predictors].values.astype(bool), distance)
            print('-- Pairwise distance computation took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())
            with open(dist_fname, 'wb') as pyt:
                joblib.dump(dist, pyt)

        ward_linkage = hierarchy.ward(dist)
        tree = hierarchy.to_tree(ward_linkage)
        cluster_colname = 'cluster_%s' % distance
        df[cluster_colname] = hierarchy.fcluster(
            ward_linkage,
            _get_height_at(tree, n_clusters),
            criterion="distance")
        print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())
        CATEGORICAL_FEATURES.append(cluster_colname)
        sns.catplot(x=cluster_colname, y=PREDICTAND, data=df.groupby('isTrain').get_group(True), kind="violin")
        plt.savefig(os.path.join(ANALYSIS_PATH, '%s.png' % cluster_colname))
        plt.close()

    # Keep euclidean clusters as 'cluster_colname' for K-fold grouping
    cluster_colname = "cluster_euclidean"

    print('Mojena stopping rule')

    clusters_for_plot = np.arange(1, 101)
    heights = np.array([_get_height_at(tree, n_clusters) for n_clusters in clusters_for_plot])

    plt.figure()
    plt.plot(clusters_for_plot, heights, 'ko--')
    plt.grid()
    plt.xlabel('Number of clusters')
    plt.ylabel('Dendrogram height')
    plt.savefig(os.path.join(ANALYSIS_PATH, '%s_mojena.png' % cluster_colname))
    plt.close()

    print('Dendrogram for Euclidean distance')
    dn = hierarchy.dendrogram(ward_linkage, no_labels=True, above_threshold_color='k')
    plt.ylabel('height')
    plt.xlabel('samples')
    plt.savefig(os.path.join(ANALYSIS_PATH, '%s_dendrogram.png' % cluster_colname))
    plt.close()

    df[predictors] = np.log1p(df[predictors])

    st_time = datetime.datetime.now()
    def func_agg(row):
        r = row[row > 0]
        return np.append([
            (row > 0).sum(),
            r.mean(),
            (r**2).mean(),
            r.std(),
            r.max(),
            r.min(),
            r.skew(),
            r.kurtosis(),
            ], r.quantile(q=percs/100))
    print('Computing non-zero aggregates...')
    df[[
        'count_nonzero',
        'mean_nonzero',
        'meansq_nonzero',
        'std_nonzero',
        'max_nonzero',
        'min_nonzero',
        'skew_nonzero',
        'kurt_nonzero',
        ] + ['p%i' % p for p in percs]] = df[predictors].apply(
            func_agg, axis=1, result_type="expand").fillna(0)
    print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())

    st_time = datetime.datetime.now()
    def func_agg(row):
        r = row[row > 0].diff().abs()
        return np.append([
            r.mean(),
            (r**2).mean(),
            r.std(),
            r.max(),
            r.min(),
            r.skew(),
            r.kurtosis(),
            ], r.quantile(q=percs/100))
    print('Computing diff aggregates...')
    df[[
        'diff_mean_nonzero',
        'diff_meansq_nonzero',
        'diff_std_nonzero',
        'diff_max_nonzero',
        'diff_min_nonzero',
        'diff_skew_nonzero',
        'diff_kurtosis_nonzero',
        ] + ['diff_p%i' % p for p in percs]] = df[predictors].apply(
            func_agg, axis=1, result_type="expand").fillna(0)
    print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())


    # add occurrences (will it help ?)
    print('Computing distributions...')
    def func_epd(row):
        epd = np.histogram(np.exp(row[row > 0].values) - 1, bins=bins_target, normed=True)[0]
        return epd / np.sum(epd)
    df[['epd_%i' % b for b in bins_target[:-1]]] = df[predictors].apply(func_epd, axis=1, result_type="expand").fillna(0)

    columns_now = df.columns
    calculated_cols.extend([c for c in columns_now if c not in columns_then])

    ## Scale the features
    #st_time = datetime.datetime.now()
    #print('Scaling (log) the features')
    #for col in df.columns:
        #if col not in [PREDICTAND, ID, 'isTrain']:
            #df[col] = np.log(df[col] + 1)
    #print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())

    df.drop(predictors, axis=1, inplace=True)
    predictors = [c for c in calculated_cols if c in df.columns]

    print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())

    #with open(os.path.join(TRAIN_PATH, 'predictors_%s.pyt' % datetime.datetime.now().strftime('%Y%m%d%H')),
              #'wb') as pyt:
        #joblib.dump(df, pyt)

    st_time = datetime.datetime.now()
    print('Transforming the features')
    cols_to_remove = []
    cols_to_add = []
    for col in predictors:
        if col in CATEGORICAL_FEATURES:
            print('\tFeature %s is categorical -> OneHot' % col)
            transf = OneHotEncoder()
            transf.fit(df[col].values.reshape(-1, 1))

            res = transf.transform(df[col].values.reshape(-1, 1))
            for i, ax in enumerate(res.transpose(), 1):
                onehot = '{}_{}'.format(col, i)
                df[onehot] = ax.toarray().squeeze()
                cols_to_add.append(onehot)
            cols_to_remove.append(col)

        else:
            print('\tFeature %s is numerical -> QuantileTransformer' % col)
            try:
                df[col] = QuantileTransformer().fit_transform(df[col].values.reshape(-1, 1))
            except:
                print("\033[91mQuantileTransformer failed on %s\033[0m" % col)
    print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())

    #df.drop(cols_to_remove, axis=1, inplace=True)
    IGNORE_FEATURES.extend(cols_to_remove)
    for col in cols_to_remove:
        predictors.remove(col)
        calculated_cols.remove(col)
    predictors.extend(cols_to_add)

    # T-SNE
    st_time = datetime.datetime.now()
    print('Running T-SNE...')
    fname = os.path.join(ANALYSIS_PATH, "tsne", "tsne.png")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    tsne_comps = tsne(
        df[predictors + [PREDICTAND, 'isTrain']], fname,
        nb=len(df),
        perplexity=40, title=None,
        visu_tsne=None, cmap='viridis',
        predictand=PREDICTAND, binary=False,
        #do_not_plot=[c for c in predictors if not c in calculated_cols + ['isTrain', PREDICTAND]],
        )

    with open(os.path.join(TRAIN_PATH, "tsne_%s.pyt" % (datetime.datetime.now().strftime('%Y%m%d%H%M'))), 'wb') as pyt:
        joblib.dump(tsne_comps, pyt)

    try:
        for i, tsne_ax in enumerate(tsne_comps.transpose(), 1):
            df['tsne%i' % i] = tsne_ax
            calculated_cols.append('tsne%i' % i)
    except:
        print('\033[91mWARNING ! could not add t-sne values\033[0m')
        print_exc()
        pass
    print('-- Took %i seconds.' % (datetime.datetime.now() - st_time).total_seconds())

    analyze(df, calculated_cols, step='preprocessed')
    #analyze_bivariate(df, cfgs, step='preprocessed')

    df_train = select_sample(df, "train")
    df_test = select_sample(df, "test")

    fname = os.path.join(TRAIN_PATH, 'df_train.pyt')
    print('Saving df_train to \033[92m%s\033[0m' % fname)
    with open(fname, 'wb') as pyt:
        joblib.dump(df_train, pyt)

    predictors = [c for c in df_train.columns if c not in IGNORE_FEATURES]
    predictors.remove(PREDICTAND)


    X_train = df_train[predictors].values
    y_train = df_train[PREDICTAND].values
    X_test = df_test[predictors].values
    test_rows = df_test.index


    # Load the "leaked" target
    leaked_target = df_leaked.loc[test_rows, leak_cols].median(axis=1)
    leaked_count = df_leaked.loc[test_rows, leak_cols].notnull().sum(axis=1)
    leak_inds = np.where(leaked_count > 0)[0]

    #reg, _ = train_and_validate(
        #df_train,
        #predictors,
        #PREDICTAND,
        #wdir=TRAIN_PATH,
        #kind='regression',
        #MLP_options={'hidden_layer_sizes': (100, 100)},
        #GradientBoosting_options={'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 600, 'random_state': 42},
        #XGBoost_options={'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 600, 'random_state': 42},
        #LightGBM_options={'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 600, 'random_state': 42, 'verbose': -1, 'num_leaves': 124},
        #RandomForest_options={'max_depth': None, 'n_estimators': 900, 'max_features': 1, 'min_samples_leaf': 3, 'min_samples_split': 10, 'criterion': 'mse', 'random_state': 42},
        #)

    #os.makedirs(OUTPUT_PATH, exist_ok=True)
    #for name, regdict in reg.items():
        #model = regdict['model']
        #fname = os.path.join(OUTPUT_PATH, '%s.csv' % name)
        #y_pred = model.predict(X_test)
        #y_pred = np.expm1(y_pred)

        ##y_pred[leak_inds] = leaked_target.values[leak_inds]

        #df_result = pd.DataFrame({ID: df_test.index,
                                  #PREDICTAND: y_pred})
        #df_result.to_csv(fname, index=False)
        #print('Wrote prediction file: \033[94;1m%s\033[0m' % fname)



    def save_model(model, name, y_pred=None, replace_leak=False):
        if model is not None:
            fname = os.path.join(TRAIN_PATH, "%s.pyt" % name)
            os.makedirs(TRAIN_PATH, exist_ok=True)
            with open(fname, "wb") as pyt:
                joblib.dump({'model': model}, pyt)
            print('\tSaved model to \033[92m%s\033[0m' % fname)

        fname = os.path.join(OUTPUT_PATH, "%s.csv" % name)
        if y_pred is None:
            y_pred = model.predict(X_test)
        y_pred = np.expm1(y_pred)

        if replace_leak:
            y_pred[leak_inds] = leaked_target.values[leak_inds]
            fname = fname.replace('.csv', '_leak.csv')

        df_result = pd.DataFrame({ID: df_test.index,
                                  PREDICTAND: y_pred})
        df_result.to_csv(fname, index=False)
        print('\tSaved prediction to \033[92m%s\033[0m' % fname)


    from lightgbm import Dataset
    from lightgbm import train as train_lgb

    nfolds = 10
    #folds = KFold(n_splits=nfolds, shuffle=True, random_state=21)
    folds = GroupKFold(n_splits=nfolds)

    y_pred_xgb = np.zeros(len(X_test))
    y_train_xgb = np.zeros(len(X_train))
    y_pred_lgbm = np.zeros(len(X_test))
    y_train_lgbm = np.zeros(len(X_train))

    lgb_params = {
        'task':'train',
        'boosting_type':'gbdt',
        'objective': 'regression',
        'metric': {'mse'},
        'num_leaves': 124,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1,
        'num_boost_round': 15000,
        'early_stopping_rounds': 100,
        'nthread': 26}

    def _rmse_func(predictions, ground_truth):
        return np.sqrt(mean_squared_error(predictions, ground_truth))

    def rmse(predictions, train_data):
        labels = train_data.get_label()
        return 'RMSE', _rmse_func(predictions, labels), False


    for ifold, (trn_idx, val_idx) in enumerate(folds.split(
        X_train,
        y_train,
        df_train[cluster_colname].values)):

        print("Fold nb. %i" % ifold)

        lgb_train = Dataset(
            data=X_train[trn_idx, :],
            label=y_train[trn_idx],
            feature_name=predictors)

        lgb_val = Dataset(
            data=X_train[val_idx, :],
            label=y_train[val_idx],
            feature_name=predictors)

        reg = XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            random_state=42)
        reg.fit(df_train[predictors].iloc[trn_idx,:].values,
                df_train[[PREDICTAND]].iloc[trn_idx, :].values.squeeze())
        pred_fold = reg.predict(df_train[predictors].iloc[val_idx].values)

        print('\t[XGBoost] oof RMSE is: \033[92m%.4f\033[0m' % np.sqrt(mean_squared_error(
            df_train[[PREDICTAND]].iloc[val_idx].values.squeeze(), pred_fold)))
        y_train_xgb += reg.predict(X_train) / nfolds
        y_pred_xgb += reg.predict(X_test) / nfolds

        reg = train_lgb(
            lgb_params,
            lgb_train,
            num_boost_round=15000,
            early_stopping_rounds=100,
            verbose_eval=100,
            valid_sets=[lgb_train, lgb_val],
            feval=rmse)

        y_pred = reg.predict(X_train[val_idx, :], num_iteration=reg.best_iteration)
        score = np.sqrt(mean_squared_error(y_train[val_idx], y_pred))

        print('\t[LGBM] Best iteration: \033[92m%i\033[0m' % reg.best_iteration)
        print('\t[LGBM] oof RMSE is: \033[92m%.4f\033[0m' % score)

        y_train_lgbm += reg.predict(X_train, num_iteration=reg.best_iteration) / nfolds
        y_pred_lgbm += reg.predict(X_test, num_iteration=reg.best_iteration) / nfolds

    save_model(None, "LightGBM_folded", y_pred_lgbm, replace_leak=True)

    save_model(None, "XGBoost_folded", y_pred_xgb)
    save_model(None, "LightGBM_folded", y_pred_lgbm)
    save_model(None, "XGB-LGBM_folded", 0.5 * (y_pred_xgb + y_pred_lgbm))

    gsDict = {}

    ## AdaBoost
    #print('\033[1mGridSearch - AdaBoostRegressor\033[0m')
    #reg_base = DecisionTreeRegressor()
    #reg = AdaBoostRegressor(reg_base, random_state=42)
    #ada_param_grid = {
        #"base_estimator__criterion": ["mse", "mae"],
        #"base_estimator__splitter": ["best", "random"],
        #"algorithm": ["SAMME", "SAMME.R"],
        #"n_estimators": [2, 10, 50],
        #"learning_rate":  [0.001, 0.01, 0.1]}

    #gsAdaBoost = GridSearchCV(reg, param_grid=ada_param_grid,
                              #cv=nfolds, scoring="neg_mean_squared_error",
                              #n_jobs=20, verbose=1)
    #gsAdaBoost.fit(X_train, y_train)

    #ada_best = gsAdaBoost.best_estimator_
    #print('\tBest score: \033[92m%.4f\033[0m' % gsAdaBoost.best_score_)
    #ada_best.fit(X_train, y_train)
    #save_model(ada_best, "AdaBoost")
    #gsDict["AdaBoost"] = gsAdaBoost


    ## ExtraTrees
    #print('\033[1mGridSearch - ExtraTreesRegressor\033[0m')
    #reg = ExtraTreesRegressor()

    ## Search grid for optimal parameters
    #ex_param_grid = {
        #"max_depth": [None],
        #"max_features": [1, 3, 10],
        #"min_samples_split": [2, 3, 10],
        #"min_samples_leaf": [1, 3, 10],
        #"bootstrap": [False],
        #"n_estimators": [100, 300, 900],
        #"criterion": ["mse", "mae"]}

    #gsExtraTrees = GridSearchCV(reg, param_grid=ex_param_grid,
                                #cv=nfolds, scoring="neg_mean_squared_error",
                                #n_jobs=20, verbose=1)
    #gsExtraTrees.fit(X_train, y_train)
    #etc_best = gsExtraTrees.best_estimator_
    #print('\tBest score: \033[92m%.4f\033[0m' % gsExtraTrees.best_score_)
    #etc_best.fit(X_train, y_train)
    #save_model(etc_best, "ExtraTrees")
    #gsDict["ExtraTrees"] = gsExtraTrees

    ## RF Parameters
    #print('\033[1mGridSearch - RandomForestRegressor\033[0m')
    #reg = RandomForestRegressor()

    ## Search grid for optimal parameters
    #rf_param_grid = {
        #"max_depth": [None, 4, 5],
        #"max_features": [1, 3, 10],
        #"min_samples_split": [2, 3, 10],
        #"min_samples_leaf": [1, 3, 10],
        #"bootstrap": [False],
        #"n_estimators": [100, 300, 900],
        #"criterion": ["mse", "mae"]}

    #gsRandomForest = GridSearchCV(
        #reg, param_grid=rf_param_grid,
        #cv=nfolds, scoring="neg_mean_squared_error",
        #n_jobs=36, verbose=1)
    #gsRandomForest.fit(X_train, y_train)
    #rfc_best = gsRandomForest.best_estimator_
    #print('\tBest score: \033[92m%.4f\033[0m' % gsRandomForest.best_score_)
    #for key in rf_param_grid.keys():
        #print('\t\t%s: \033[92m%s\033[0m' % (key, getattr(rfc_best, key, '-')))

    #rfc_best.fit(X_train, y_train)
    #save_model(rfc_best, "RandomForest")
    #gsDict["RandomForest"] = gsRandomForest

    ## Gradient boosting
    #print('\033[1mGridSearch - GradientBoostingRegressor\033[0m')
    #reg = GradientBoostingRegressor()
    #gb_param_grid = {
        #'loss' : ["ls", "lad", "huber"],
        #'n_estimators' : [600, 300, 900],
        #'learning_rate': [0.1, 0.05, 0.01],
        #'max_depth': [5, 4, 6],
        #'min_samples_leaf': [10, 50],
        #'max_features': ["sqrt", "auto"]
        #}

    #gsGradientBoosting = GridSearchCV(
        #reg, param_grid=gb_param_grid,
        #cv=nfolds, scoring="neg_mean_squared_error",
        #n_jobs=36, verbose=1)
    #gsGradientBoosting.fit(X_train, y_train)
    #gbc_best = gsGradientBoosting.best_estimator_
    #print('\tBest score: \033[92m%.4f\033[0m' % gsGradientBoosting.best_score_)
    #for key in gb_param_grid.keys():
        #print('\t\t%s: \033[92m%s\033[0m' % (key, getattr(gbc_best, key, '-')))

    #gbc_best.fit(X_train, y_train)
    #save_model(gbc_best, "GradientBoosting")
    #gsDict["GradientBoosting"] = gsGradientBoosting

    # Gradient boosting
    print('\033[1mGridSearch - XGBRegressor\033[0m')
    reg = XGBRegressor()
    xgb_param_grid = {
        'n_estimators' : [600, 300, 900],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [5, 4, 6],
        'missing': [None, 0.],
        'booster': ["gbtree", "gblinear", "dart"],
        }

    gsXGBoost = GridSearchCV(
        reg, param_grid=xgb_param_grid,
        cv=nfolds, scoring="neg_mean_squared_error",
        n_jobs=36, verbose=1)
    gsXGBoost.fit(X_train, y_train)
    gbc_best = gsXGBoost.best_estimator_
    print('\tBest score: \033[92m%.4f\033[0m' % gsXGBoost.best_score_)
    for key in xgb_param_grid.keys():
        print('\t\t%s: \033[92m%s\033[0m' % (key, getattr(gbc_best, key, '-')))

    gbc_best.fit(X_train, y_train)
    save_model(gbc_best, "XGBoost")
    gsDict["XGBoost"] = gsXGBoost

    # TODO GridSearch for LightGBM !!!!!!!!!


if __name__ == '__main__':
    import sys
    main("-v" in sys.argv, test='-t' in sys.argv)
