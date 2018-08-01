#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Tools for sample analysis & visualization

'''

import os
import joblib
import datetime

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.colors import rgb2hex
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

from cycler import cycler
from netCDF4 import num2date

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, SpectralEmbedding


# COLOR_HUE -> obtained with "I want hue" (tools.medialab.sciences-po.fr/iwanthue/)
COLOR_HUE = np.array([
    [94, 107, 219],
    [208, 82, 47],
    [176, 179, 55],
    [165, 89, 200],
    [94, 182, 78],
    [202, 70, 160],
    [85, 183, 147],
    [212, 58, 96],
    [82, 123, 56],
    [215, 133, 196],
    [219, 146, 52],
    [183, 171, 102],
    [89, 104, 174],
    [161, 147, 222],
    [140, 109, 41],
    [79, 170, 217],
    [161, 78, 59],
    [158, 75, 115],
    [223, 140, 103],
    [223, 121, 139]]
).astype(float) / 256.

FONTSIZE = 22
PREDICTAND_CMAP = LinearSegmentedColormap.from_list('predictand', [COLOR_HUE[0], COLOR_HUE[1]], N=2)



def univariate_sample_analysis(df, directory, verbose=False, **kwargs):
    '''Analyses the features of a DataFrame, one by one
    depending on their type (numerical, categorical...)

    kwargs:
        - categorical_features
        - predictand
        - labels, titles... (cf other functions)
    '''
    id_features = kwargs.get('id_features', [])
    categorical_features = kwargs.get('categorical_features', [])
    predictand = kwargs.get('predictand', 'churn')
    encoders = kwargs.get('encoders', {})

    print('Univariate analysis')
    os.makedirs(directory, exist_ok=True)
    for col in df.columns:

        if col in id_features:
            continue

        kwargs_func = kwargs.copy()
        kwargs_func['encoder'] = encoders.get(col, None)

        if col == predictand:
            print('\t\033[93m%s [predictand, %s]\033[0m' % (predictand, col))
            univariate_predictand_analysis(df, directory, **kwargs_func)

        elif col in categorical_features:
            print('\t\033[92m%s [categorical]\033[0m' % col)
            univariate_category_analysis(df, col, directory, **kwargs_func)

        else:
            print('\t\033[91m%s [numerical]\033[0m' % col)
            univariate_numerical_analysis(df, col, directory, **kwargs_func)


def bivariate_sample_analysis(df, directory, verbose=False, **kwargs):
    '''Analyses the features of a DataFrame, two by two
    depending on their type (numerical, categorical...)

    kwargs:
        - categorical_features
        - predictand
        - labels, titles... (cf other functions)
    '''
    from itertools import combinations

    id_features = kwargs.get('id_features', [])
    categorical_features = kwargs.get('categorical_features', [])
    predictand = kwargs.get('predictand', 'churn')
    encoders = kwargs.get('encoders', {})

    # TODO ? use SelectKBest (+ retrieve the names of the selected features...) if there are too many combinations ?

    pairs = combinations([c for c in df.columns if c not in id_features + [predictand]], 2)

    print('Bivariate analysis')
    os.makedirs(directory, exist_ok=True)
    for (col_x, col_y) in pairs:
        print('\tCombination: %s, %s' % (col_x, col_y))
        kwargs_func = kwargs.copy()


        if col_y in categorical_features:
            if col_x in categorical_features:
                kwargs_func['encoder_x'] = encoders.get(col_x, None)
                kwargs_func['encoder_y'] = encoders.get(col_y, None)
                bivariate_categories_analysis(df, col_x, col_y, directory, **kwargs_func)

            else:
                col_x, col_y = col_y, col_x
                kwargs_func['encoder_x'] = encoders.get(col_x, None)
                bivariate_category_numerical_analysis(df, col_x, col_y, directory, **kwargs_func)

        else:
            if col_x in categorical_features:
                kwargs_func['encoder_x'] = encoders.get(col_x, None)
                bivariate_category_numerical_analysis(df, col_x, col_y, directory, **kwargs_func)

            else:
                bivariate_numericals_analysis(df, col_x, col_y, directory, **kwargs_func)



def univariate_predictand_analysis(df, directory, **kwargs):
    '''Not interesting in case of a binary predictand
    TODO adapt to non-binary predictand
    '''
    return


def univariate_category_analysis(df, column, directory, **kwargs):
    '''Histograms: total sample vs. positive sample vs. negative sample
    TODO adapt to non-binary predictand

    kwargs:
        - predictand
        - label_true, label_false
        - column_labels
        - title
        - encoder
    '''
    predictand = kwargs.get('predictand', 'churn')
    label_true = kwargs.get('label_true', '%s true' % predictand)
    label_false = kwargs.get('label_false', '%s false' % predictand)
    column_labels = kwargs.get('column_labels', {})
    title = kwargs.get('title', 'Distribution')
    encoder = kwargs.get('encoder', None)

    fig, axs = plt.subplots(figsize=(14, 16), nrows=2)

    plt.sca(axs[0])

    unique = df[column].unique()
    unique = np.array(sorted(unique))

    dfg = df.groupby(predictand)

    values = df[column]
    count = np.array([
        np.sum(values == v) for v in unique], dtype=float)
    count /= np.sum(count)

    p = df[predictand]
    mean_p = df.groupby(column)[predictand].mean()
    all_mean = df[predictand].mean()


    values = dfg.get_group(0)[column]
    count_nc = np.array([
        np.sum(values == v) for v in unique], dtype=float)
    count_nc /= np.sum(count_nc)

    values = dfg.get_group(1)[column]
    count_c = np.array([
        np.sum(values == v) for v in unique], dtype=float)
    count_c /= np.sum(count_c)

    w = 0.3
    plt.bar(
        unique - (w / 2), count, w,
        color=rgb2hex(COLOR_HUE[2]),
        edgecolor=rgb2hex(COLOR_HUE[2]),
        )

    plt.bar(
        unique - (w * 1.5), count_nc, w,
        color=rgb2hex(COLOR_HUE[0]),
        edgecolor=rgb2hex(COLOR_HUE[0]),
        )
    plt.bar(
        unique + (w * 0.5), count_c, w,
        color=rgb2hex(COLOR_HUE[1]),
        edgecolor=rgb2hex(COLOR_HUE[1]),
        )

    plt.legend(['total', label_false, label_true], loc='best', fontsize=FONTSIZE)

    #plt.xlabel('%s category' % CUTE_LABELS.get(column, column), fontsize=FONTSIZE)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(title, fontsize=FONTSIZE)
    if encoder is None:
        plt.xticks(unique, unique, fontsize=FONTSIZE)
    else:
        plt.xticks(unique, encoder.inverse_transform(unique), fontsize=FONTSIZE, rotation=90) # [encoder.classes_[u] for u in unique]
    plt.yticks(fontsize=FONTSIZE)
    plt.grid()
    plt.tight_layout()


    plt.sca(axs[1])
    plt.bar(
        unique - 0.3, np.ones_like(count_nc), 0.6,
        color=rgb2hex(COLOR_HUE[0]),
        edgecolor=rgb2hex(COLOR_HUE[0]),
        )
    plt.bar(
        unique - 0.3, mean_p, 0.6,
        color=rgb2hex(COLOR_HUE[1]),
        edgecolor=rgb2hex(COLOR_HUE[1]),
        )

    plt.plot(plt.xlim(), [all_mean, all_mean], 'k-')

    plt.ylim(0, 1)

    plt.legend(['mean %s' % predictand, label_false, label_true], loc='best', fontsize=FONTSIZE)
    plt.xlabel('%s category' % column_labels.get(column, column), fontsize=FONTSIZE)
    plt.ylabel('')
    plt.title('%s rate for each %s category' % (predictand, column_labels.get(column, column.lower())), fontsize=FONTSIZE)
    if encoder is None:
        plt.xticks(unique, unique, fontsize=FONTSIZE)
    else:
        plt.xticks(unique, encoder.inverse_transform(unique), fontsize=FONTSIZE, rotation=90)
    plt.yticks(fontsize=FONTSIZE)
    plt.tight_layout()

    fname = os.path.join(directory, '%s.png' % column)
    plt.savefig(fname)
    print('\t\tFigure \033[92m%s\033[0m -- ok.' % fname)
    plt.close()


def univariate_numerical_analysis(df, column, directory, **kwargs):
    '''Boxplots: total sample vs. positive sample vs. negative sample'''
    predictand = kwargs.get('predictand', 'churn')
    label_true = kwargs.get('label_true', '%s true' % predictand)
    label_false = kwargs.get('label_false', '%s false' % predictand)
    column_labels = kwargs.get('column_labels', {})
    title = kwargs.get('title', 'Distribution')

    #fig, axs = plt.subplots(figsize=(14, 14), nrows=2)
    # WARNING
    fig, axs = plt.subplots(figsize=(14, 21), nrows=3)

    dfg = df.groupby(predictand)

    df_nonan = df[[column, predictand]].dropna()
    values = df_nonan[column].values

    series_true = dfg.get_group(1)[column].dropna()
    series_false = dfg.get_group(0)[column].dropna()

    plt.sca(axs[0])
    plt.boxplot(
        (
            values,
            series_true,
            series_false,
        ),
        capprops={'linewidth': 1, 'color': COLOR_HUE[0], 'alpha': 0.9},
        boxprops={'linewidth': 3, 'color': COLOR_HUE[0], 'alpha': 0.9},
        whiskerprops={'linewidth': 1, 'color': COLOR_HUE[0], 'alpha': 0.9},
        medianprops={'linewidth': 1, 'color': COLOR_HUE[1], 'alpha': 0.9},
        )

    if column.startswith('date_'):
        yticks, _ = plt.yticks()
        dates = get_dates(yticks)
        plt.yticks(yticks, [d.strftime('%Y%m') for d in dates])

    plt.yticks(fontsize=FONTSIZE - 2)
    plt.xticks(np.arange(3) + 1, ['total',
                                  label_true,
                                  label_false], fontsize=FONTSIZE - 2)

    plt.title('Boxplots of %s' % column_labels.get(column, column), fontsize=FONTSIZE)
    plt.ylabel('')
    plt.xlabel('')

    plt.grid()
    plt.tight_layout()

    count_false, edges_false = np.histogram(series_false, bins=40, normed=True)
    count_true, edges_true = np.histogram(series_true, bins=edges_false, normed=True)
    count_false *= 100. / np.sum(count_false)
    count_true *= 100. / np.sum(count_true)

    width = np.diff(edges_false)[0]/2
    ww = width / 3

    plt.sca(axs[1])
    plt.bar(
        edges_false[:-1] + ww,
        count_false,
        width=width - ww,
        color=COLOR_HUE[0],
        )
    plt.bar(
        edges_true[:-1] + width,
        count_true,
        width=width - ww,
        color=COLOR_HUE[1],
        )

    plt.grid()
    plt.yticks(fontsize=FONTSIZE - 2)

    if column.startswith('date_'):
        xticks, _ = plt.xticks()
        dates = get_dates(xticks)
        plt.xticks(xticks, [d.strftime('%Y%m') for d in dates], rotation=90)

    plt.xticks(fontsize=FONTSIZE - 2)

    plt.title('Distribution per {}'.format(column_labels.get(column, column)), fontsize=FONTSIZE)
    plt.xlabel(column_labels.get(column, column), fontsize=FONTSIZE)
    plt.ylabel('')

    plt.legend([label_false, label_true], loc='best', fontsize=FONTSIZE)
    plt.tight_layout()

    plt.sca(axs[2])
    plt.bar(
        edges_false[:-1],
        np.ones_like(count_false),
        width=width,
        color=COLOR_HUE[0],
        )
    plt.bar(
        edges_false[:-1],
        count_true / (count_true + count_false),
        width=width,
        color=COLOR_HUE[1],
        )
    plt.plot(plt.xlim(), [0.5, 0.5], 'k-')
    plt.ylim(0, 1)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.xlabel(column_labels.get(column, column), fontsize=FONTSIZE)
    plt.ylabel('')
    plt.title('{} rate per {}'.format(predictand, column_labels.get(column, column)), fontsize=FONTSIZE)

    plt.legend(['50% rate', label_false, label_true], loc='best', fontsize=FONTSIZE)
    plt.tight_layout()
    # TODO same bars as the bottom graph of categorical features

    fname = os.path.join(directory, '%s.png' % column)
    plt.savefig(fname)
    print('\t\tFigure \033[91m%s\033[0m -- ok.' % fname)
    plt.close()



def bivariate_categories_analysis(df, col_x, col_y, directory, **kwargs):
    ''''''
    unique = np.sort(df[col_x].unique()).astype(float)
    bins = unique - np.append(np.diff(unique) / 2, np.diff(unique)[-1] / 2)
    bins = np.append(bins, 2 * bins[-1] - bins[-2])
    kwargs['bins_x'] = np.unique(bins)

    unique = np.sort(df[col_y].unique()).astype(float)
    bins = unique - np.append(np.diff(unique) / 2, np.diff(unique)[-1] / 2)
    bins = np.append(bins, 2 * bins[-1] - bins[-2])
    kwargs['bins_y'] = np.unique(bins)
    return _bivariate_analysis(df, col_x, col_y, directory, **kwargs)

def bivariate_category_numerical_analysis(df, col_x, col_y, directory, **kwargs):
    ''''''
    unique = np.sort(df[col_x].unique()).astype(float)
    bins = unique - np.append(np.diff(unique) / 2, np.diff(unique)[-1] / 2)
    bins = np.append(bins, 2 * bins[-1] - bins[-2])
    kwargs['bins_x'] = np.unique(bins)
    return _bivariate_analysis(df, col_x, col_y, directory, **kwargs)

def bivariate_numericals_analysis(df, col_x, col_y, directory, **kwargs):
    ''''''
    return _bivariate_analysis(df, col_x, col_y, directory, **kwargs)

def _bivariate_analysis(df, col_x, col_y, directory, **kwargs):
    predictand = kwargs.get('predictand', 'churn')
    label_true = kwargs.get('label_true', '%s true' % predictand)
    label_false = kwargs.get('label_false', '%s false' % predictand)
    column_labels = kwargs.get('column_labels', {})

    encoder_x = kwargs.get('encoder_x', None)
    encoder_y = kwargs.get('encoder_y', None)

    bins_x = kwargs.get('bins_x', 100)
    bins_y = kwargs.get('bins_y', 100)

    fig, axs = plt.subplots(figsize=(14, 14), nrows=2)

    plt.sca(axs[0])
    plt.scatter(df[col_x].values.ravel(),
                df[col_y].values.ravel(),
                c=df[predictand].values.ravel(),
                edgecolors='face',
                alpha=0.8,
                cmap=cm.viridis)

    if encoder_x is None:
        plt.xticks(fontsize=FONTSIZE - 2)
    else:
        unique = df[col_x].unique()
        plt.xticks(unique, encoder_x.inverse_transform(unique), fontsize=FONTSIZE - 2) # [encoder_x.classes_[u] for u in unique]

    if encoder_y is None:
        plt.yticks(fontsize=FONTSIZE - 2)
    else:
        unique = df[col_y].unique()
        plt.yticks(unique, encoder_y.inverse_transform(unique), fontsize=FONTSIZE - 2)

    plt.xlabel(column_labels.get(col_x, col_x), fontsize=FONTSIZE)
    plt.ylabel(column_labels.get(col_y, col_y), fontsize=FONTSIZE)
    plt.title('{} vs. {}'.format(col_x, col_y), fontsize=FONTSIZE)
    cbar = plt.colorbar()
    cbar.set_label(label_true)
    plt.grid()
    plt.tight_layout()

    df_ = df[[col_x, col_y]].dropna()

    plt.sca(axs[1])
    plt.hist2d(df_[col_x].values.ravel(),
               df_[col_y].values.ravel(),
               cmap=cm.viridis,
               bins=(bins_x, bins_y),
               )

    if encoder_x is None:
        plt.xticks(fontsize=FONTSIZE - 2)
    else:
        unique = df[col_x].unique()
        plt.xticks(unique, [encoder_x.classes_[u] for u in unique], fontsize=FONTSIZE - 2)

    if encoder_y is None:
        plt.yticks(fontsize=FONTSIZE - 2)
    else:
        unique = df[col_y].unique()
        plt.yticks(unique, [encoder_y.classes_[u] for u in unique], fontsize=FONTSIZE - 2)

    plt.xlabel(column_labels.get(col_x, col_x), fontsize=FONTSIZE)
    plt.ylabel(column_labels.get(col_y, col_y), fontsize=FONTSIZE)
    plt.title('{} vs. {}'.format(col_x, col_y), fontsize=FONTSIZE)
    cbar = plt.colorbar()
    cbar.set_label('Distribution')
    plt.tight_layout()

    fname = os.path.join(directory, '%s_%s.png' % (col_x, col_y))
    plt.savefig(fname)
    print('\t\tFigure \033[32m%s\033[0m -- ok.' % fname)
    plt.close()


def _scatter_plot(fname, data, c_col=None, col=None, cmap='magma', title=None, log_c=False):
    '''Plots a scatter figure (based on the provided data components)
    i.e. t-SNE components, PCA or MDS...
    can use a column to colorize the samples
    '''
    plt.figure(figsize=(14, 14))

    if c_col is None:
        plt.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.6,
            s=40,
            edgecolors='face',
            )
    else:
        if log_c:
            col[col < 0] = 0
            col = np.log(col + 1)

        plt.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.6,
            s=50,
            edgecolors='face',
            c=col,
            cmap=cmap,
            )
        cbar = plt.colorbar()
        cbar.set_label(c_col, size=FONTSIZE)
        cbar.ax.tick_params(labelsize=FONTSIZE)

    if title is not None:
        plt.title(title, fontsize=FONTSIZE + 2)

    plt.axis('off')
    plt.savefig(fname)
    plt.close()


def plot_pca(df, fname, c_col='churn', annotate=None, debug=False, remove_c_col=True, npc=6):
    '''Principal component analysis & graphical representation
    '''
    dfn = df.copy()
    col = dfn[c_col].as_matrix()
    if remove_c_col: del dfn[c_col]
    dfn = dfn.dropna()

    print('\tPerforming a PCA with %s components (%s features in the input data)' % (npc, dfn.shape[1]))
    pca = PCA(n_components=npc).fit(dfn.values)
    pcs = pca.transform(dfn.values)
    print('\tExplained variance: %.2f percent' % (np.sum(pca.explained_variance_ratio_) * 100))
    #print('\tNb. of NaNs in pcs: %s' % (np.sum(np.isnan(pcs))))
    eofs = pca.components_

    nrows, ncols = 2, 3

    fig, axs = plt.subplots(figsize=(26, 24), nrows=nrows, ncols=ncols)

    for k in range(pcs.shape[1] - 1):
        i = k // ncols
        j = k % ncols
        plt.sca(axs[i, j])

        plt.scatter(pcs[:, 0], pcs[:, k + 1], c=col,
                    s=30, alpha=0.8, cmap='viridis', edgecolors="face")
        cbar = plt.colorbar()
        cbar.set_label(c_col)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA %i' % (k + 2))
        plt.grid()

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname)
    print('Make PCA figure: \033[92m%s\033[0m' % fname)
    plt.close()

    return pca, pcs


def tsne(df, fname, nb=1000, perplexity=30, title=None, visu_tsne=None, cmap='viridis', predictand="churn", binary=True, init="random", do_not_plot=[]):
    '''T-SNE computation & visualization with a color to highlight a specific value

    visu_tsne
    -> provide coordinates previously computed to reduce computation time
    -> needs to be on the same sample then
    '''
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    rndperm = np.random.permutation(df.shape[0])
    inds = rndperm[:nb]
    remove_cols_tsne = [predictand, 'predicted', 'isTrain']
    sample = df.loc[df.index[inds], [c for c in df.columns if c not in remove_cols_tsne]].dropna()

    if visu_tsne is None:
        visu_model = TSNE(n_components=2, perplexity=perplexity, random_state=42, init=init)
        visu_tsne = visu_model.fit_transform(
            sample.values
            )

    for c_col in df.columns:

        if c_col in do_not_plot:
            continue

        # Adapt the color map to the feature
        if c_col == 'predicted':
            cmap_visu = cmap
        else:
            cmap_visu = PREDICTAND_CMAP if (c_col in [predictand,] and binary) else 'plasma'

        _scatter_plot(
            fname.replace('.png', '_%s.png' % c_col),
            visu_tsne, c_col=c_col, col=df.loc[sample.index, c_col].as_matrix(),
            cmap=cmap_visu,
            title=title)

    return visu_tsne


def plot_features_importance(column_names, model, fname):
    plt.figure(figsize=(7, 32))
    sorted_cols = np.array(sorted(zip(column_names, model.feature_importances_), key=lambda x: -x[1]))
    sns.barplot(y=sorted_cols[:, 0], x=sorted_cols[:, 1].astype(float))
    sns.despine(left=True, bottom=False)
    print('Saving feature importances into \033[92m%s\033[0m' % fname)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
