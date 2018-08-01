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



def _scatter_plot(fname, data, feature=None, col=None, cmap='magma', title=None, log_c=False):
    '''Plots a scatter figure (based on the provided data components)
    i.e. t-SNE components, PCA or MDS...
    can use a column to colorize the samples
    '''
    plt.figure(figsize=(14, 14))

    if feature is None:
        plt.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.3,
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
            alpha=0.3,
            s=50,
            edgecolors='face',
            c=col,
            cmap=cmap,
            )
        cbar = plt.colorbar()
        cbar.set_label(feature, size=FONTSIZE)
        cbar.ax.tick_params(labelsize=FONTSIZE)

    if title is not None:
        plt.title(title, fontsize=FONTSIZE + 2)

    plt.axis('off')
    plt.savefig(fname)
    plt.close()


def plot_pca(df, fname, predictand='churn', annotate=None, debug=False, remove_predictand=True, npc=6):
    '''Principal component analysis & graphical representation
    '''
    #dfn = df.copy()
    #col = dfn[predictand].as_matrix()
    #if remove_predictand: del dfn[predictand]
    #dfn = dfn.dropna()

    X = df[[c for c in df.columns if c != predictand]].values
    y = df[predictand].values
    y[np.isnan(y)] = 0

    print('\tPerforming a PCA with %s components (%s features in the input data)' % (npc, df.shape[1]))
    pca = PCA(n_components=npc).fit(X)
    pcs = pca.transform(X)
    print('\tExplained variance: %.2f percent' % (np.sum(pca.explained_variance_ratio_) * 100))
    #print('\tNb. of NaNs in pcs: %s' % (np.sum(np.isnan(pcs))))
    eofs = pca.components_

    nrows, ncols = 2, 3

    fig, axs = plt.subplots(figsize=(26, 24), nrows=nrows, ncols=ncols)

    for k in range(pcs.shape[1] - 1):
        i = k // ncols
        j = k % ncols
        try:
            plt.sca(axs[i, j])
        except IndexError:
            break

        plt.scatter(pcs[:, 0], pcs[:, k + 1], c=y,
                    s=30, alpha=0.8, cmap='viridis', edgecolors="face")
        cbar = plt.colorbar()
        cbar.set_label(predictand)
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

    #rndperm = np.random.permutation(df.shape[0])
    #inds = rndperm[:nb]
    remove_cols_tsne = [predictand, 'predicted', 'isTrain']
    #sample = df.loc[df.index[inds], [c for c in df.columns if c not in remove_cols_tsne]].dropna()
    sample = df[[c for c in df.columns if c not in remove_cols_tsne]].fillna(0)

    if visu_tsne is None:
        visu_model = TSNE(n_components=2, perplexity=perplexity, random_state=42, init=init)
        visu_tsne = visu_model.fit_transform(
            sample.values
            )

    for predictor in df.columns:

        if predictor in do_not_plot:
            continue

        # Adapt the color map to the feature
        if predictand == 'predicted':
            cmap_visu = cmap
        else:
            cmap_visu = PREDICTAND_CMAP if (predictor in [predictand,] and binary) else 'plasma'

        _scatter_plot(
            fname.replace('.png', '_%s.png' % predictor),
            visu_tsne, feature=predictor, col=df.loc[sample.index, predictor].values,
            cmap=cmap_visu,
            title=title)

    return visu_tsne
