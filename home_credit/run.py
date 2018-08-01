import os

import joblib
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from commons.analysis import univariate_sample_analysis
from commons.transform import encode_dataframe


from sklearn.preprocessing import binarize, QuantileTransformer, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

from lightgbm import Dataset
from lightgbm import train as train_lgb
from xgboost import DMatrix
from xgboost import train as train_xgb

from traceback import print_exc

import warnings
warnings.filterwarnings("ignore")

import gc


ANALYSIS_PATH = "../analysis"
PREDICTAND = "TARGET"
CATEGORICAL_FEATURES = [
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START',
    'ORGANIZATION_TYPE',
    'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE',
    'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE',
    'LIVE_CITY_NOT_WORK_CITY',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY',
    'REG_REGION_NOT_LIVE_REGION',
    'REG_REGION_NOT_WORK_REGION',
    ]
PREVIOUS_CATEGORICAL_FEATURES = [
    'NAME_CONTRACT_TYPE',
    'WEEKDAY_APPR_PROCESS_START',
    'FLAG_LAST_APPL_PER_CONTRACT',
    'NAME_CASH_LOAN_PURPOSE',
    'NAME_CONTRACT_STATUS',
    'NAME_PAYMENT_TYPE',
    'CODE_REJECT_REASON',
    'NAME_TYPE_SUITE',
    'NAME_CLIENT_TYPE',
    'NAME_GOODS_CATEGORY',
    'NAME_PORTFOLIO',
    'NAME_PRODUCT_TYPE',
    'CHANNEL_TYPE',
    'NAME_SELLER_INDUSTRY',
    'NAME_YIELD_GROUP',
    'PRODUCT_COMBINATION',
    ]

ID_FEATURES = [
    'SK_ID_CURR',
    'SK_ID_PREV',
    ]



def analyze(df, predictors, step='raw'):

    dname = os.path.join(ANALYSIS_PATH, step)
    os.makedirs(dname, exist_ok=True)
    for col in predictors:
        print('\033[1m%s: %i missing values\033[0m [of %i]' % (col, df[col].isnull().sum(), len(df)))
        print('\tMaking figure for %s...' % col)
        fname = os.path.join(dname, "%s.png" % col)

        if col in CATEGORICAL_FEATURES:
            g = sns.catplot(PREDICTAND, col=col, col_wrap=3, data=df[df[col].notnull()], kind="count")
        else:
            g = sns.catplot(y=col, x=PREDICTAND, data=df, kind="violin")

        plt.xticks(rotation=90)
        plt.tight_layout()
        #plt.show()
        g.savefig(fname)
        print('\tSaved figure \033[92m%s\033[0m' % fname)
        plt.close()


def select_sample(df, kind="train"):
    '''kind = train or test'''
    boo = kind == "train"
    return df.groupby('isTrain').get_group(boo).drop('isTrain', axis=1)

def get_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()*100).sort_values(ascending=False)
    m = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return m[m.Total > 0]


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


def plot_feature_importances(predictors, feature_importances, dname, kind):

    def get_importance_dataframe(importances):
        df_imp = pd.DataFrame(
            {i: imp for i, imp in enumerate(importances, 1)},
            index=predictors)

        df_imp.index.name = "feature"
        df_imp = df_imp.stack().to_frame()
        return df_imp.rename(columns={df_imp.columns[0]: "importance"}).reset_index(
            ).sort_values("importance", ascending=False)

    df_imp = get_importance_dataframe(np.log1p(feature_importances))
    print(df_imp.head())

    os.makedirs(dname, exist_ok=True)
    fname = os.path.join(dname, "%s.png" % (kind,))
    plt.figure(figsize=(14, 30))
    sns.barplot(
        x="importance",
        y="feature",
        data=df_imp)
    plt.title('Feature importance (log) by %s' % kind)
    plt.tight_layout(False)
    plt.savefig(fname)
    plt.close()


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


def tsne(df, fname, nb=1000, perplexity=30, title=None, visu_tsne=None, cmap='viridis', predictand="TARGET", binary=True, init="random", do_not_plot=[]):
    '''T-SNE computation & visualization with a color to highlight a specific value

    visu_tsne
    -> provide coordinates previously computed to reduce computation time
    -> needs to be on the same sample then
    '''
    from sklearn.manifold import TSNE

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


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('../input/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_PERC'][np.logical_not(np.isfinite(ins['PAYMENT_PERC']))] = 0

    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    aggregations = {col: ['min', 'max', 'mean', 'sum', 'var'] for col in cc.columns}
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg




if __name__ == '__main__':

    # Load basic application dataframes
    df_train = pd.read_csv('../input/application_train.csv', sep=',')
    df_test = pd.read_csv('../input/application_test.csv', sep=',')

    predictors = [c for c in df_train.columns if c not in ID_FEATURES + [PREDICTAND]]
    CATEGORICAL_FEATURES.extend([c for c in df_train.columns if c.startswith('FLAG_')])


    # Join dataframes for preprocessing & feat.eng.
    df_test.index += len(df_train) + 1
    df_train['isTrain'] = True
    df_test['isTrain'] = False
    df = df_train.append(df_test)
    df.set_index("SK_ID_CURR", inplace=True)

    columns_then = df.columns


    ## Load other dataframes
    #df_previous = pd.read_csv('../input/previous_application.csv', sep=',')

    ## Correct some aberrant & missing values
    #df_previous['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    #df_previous['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    #df_previous['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    #df_previous['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    #df_previous['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    ## Add feature: value ask / value received percentage
    #df_previous['APP_CREDIT_PERC'] = df_previous['AMT_APPLICATION'] / df_previous['AMT_CREDIT']
    #df_previous['APP_CREDIT_PERC'][np.logical_not(np.isfinite(df_previous['APP_CREDIT_PERC']))] = 0

    #df_previous['DAYS_TERMINATION'][df_previous['DAYS_TERMINATION'] > 0] = np.nan
    #df_previous['NO_SELLERPLACE_AREA'] = (df_previous['SELLERPLACE_AREA'] == -1).astype(str)
    #PREVIOUS_CATEGORICAL_FEATURES.append('NO_SELLERPLACE_AREA')
    #df_previous['SELLERPLACE_AREA'].replace(-1, 0, inplace=True)
    #df_previous['SELLERPLACE_AREA'] = np.log1p(df_previous['SELLERPLACE_AREA'])


    ##dname = "../analysis/raw/previous"
    ##os.makedirs(dname, exist_ok=True)
    ##for col in df_previous.columns:
        ##if col == "SK_ID_PREV":
            ##continue
        ##elif col in PREVIOUS_CATEGORICAL_FEATURES:
            ##print('%s - countplot' % col)
            ##sns.countplot(x=col, data=df_previous)
            ##plt.xticks(rotation=90)
        ##else:
            ##print('%s - distplot' % col)
            ##sns.distplot(df_previous[col][df_previous[col].notnull()], kde=False)
            ##plt.title('%s, %i missing values' % (col, df_previous[col].isnull().sum()))

        ##plt.savefig(os.path.join(dname, "%s.png" % col))
        ##plt.close()

    #for col in df_previous.columns:
        #if df_previous[col].isnull().sum() == 0: continue
        #if col == "SK_ID_PREV":
            #continue
        #elif col in PREVIOUS_CATEGORICAL_FEATURES:
            #df_previous[col + '_nan'] = df_previous[col].isnull().astype(float)
            #df_previous[col].fillna(df_previous[col].mode()[0], inplace=True)
        #else:
            #df_previous[col + '_nan'] = df_previous[col].isnull().astype(float)
            #df_previous[col].fillna(df_previous[col].median(), inplace=True)

    #PREV_NUM_COLS = [c for c in df_previous.columns if c not in PREVIOUS_CATEGORICAL_FEATURES + ID_FEATURES]

    #print('Computing aggregates of previous applications...')
    #grp = df_previous.groupby("SK_ID_CURR")[PREV_NUM_COLS]
    #medians = grp.median()
    #means = grp.mean()
    #maxs = grp.max()
    #mins = grp.min()
    #stds = grp.std()
    #sums = grp.sum()
    #counts = grp.count()

    ## Count the number of previous applications
    #df['1PREV_app-count'] = counts.iloc[:, 0]
    #df['1PREV_no-app'] = df['1PREV_app-count'].isnull()
    #df['1PREV_app-count'].fillna(0, inplace=True)

    #for col in medians.columns:
        #if col in ID_FEATURES:
            #continue
        #elif col in PREVIOUS_CATEGORICAL_FEATURES:
            ## do stuff TODO
            #continue
        #else:
            #df['1PREV_%s-median' % col] = medians[col]
            #df['1PREV_%s-mean' % col] = means[col]
            #df['1PREV_%s-max' % col] = maxs[col]
            #df['1PREV_%s-min' % col] = mins[col]
            #df['1PREV_%s-std' % col] = stds[col]
            #df['1PREV_%s-sum' % col] = sums[col]

    ## OntHot-encode categorical features to facilitate count
    #print('\tNow on categorical features...')
    #df_prev_cat = df_previous[PREVIOUS_CATEGORICAL_FEATURES].copy()
    #for col in df_prev_cat.columns:
        #lb = LabelEncoder().fit(df_prev_cat[col].astype(str))
        #values = lb.transform(df_prev_cat[col].astype(str)).reshape(-1, 1)

        #enc = OneHotEncoder().fit(values)
        #res = enc.transform(values)

        #for i, ax in enumerate(res.transpose()):
            #ft_name = '%s_%s' % (col, lb.classes_[enc.active_features_[i]])
            #ft_name = ft_name.replace('/', '-').replace('#', '-').replace('*', '-').replace(' ', '').replace('(', '-').replace(')', '-').replace('+', '-').replace('=', '-')
            #arr = ax[0].toarray()
            #df_prev_cat[ft_name] = ax.toarray().squeeze()

    #df_prev_cat.drop(PREVIOUS_CATEGORICAL_FEATURES, axis=1, inplace=True)
    #df_prev_cat['SK_ID_CURR'] = df_previous['SK_ID_CURR']

    #cat_count = df_prev_cat.groupby('SK_ID_CURR').mean()

    #for col in cat_count.columns:
        ## .join only supports suffix...
        #df['1PREV_%s' % col] = cat_count[col]


    ### Load other dataframes
    #agg = bureau_and_balance()
    #ren = {col: "2BB_%s" % col for col in agg.columns}
    #agg.rename(columns=ren, inplace=True)
    #print("bureau_and_balance df shape:", agg.shape)
    #df = df.join(agg, how='left', on='SK_ID_CURR')

    ##univariate_sample_analysis(df[list(agg.columns) + [PREDICTAND]], "../analysis/raw/bureau_and_balance",
                               ##verbose=False,
                               ##predictand=PREDICTAND)

    #agg = pos_cash()
    #ren = {col: "3PC_%s" % col for col in agg.columns}
    #agg.rename(columns=ren, inplace=True)
    #print("pos_cash df shape:", agg.shape)
    #df = df.join(agg, how='left', on='SK_ID_CURR')

    ##univariate_sample_analysis(df[list(agg.columns) + [PREDICTAND]], "../analysis/raw/pos_cash",
                               ##verbose=False,
                               ##predictand=PREDICTAND)

    #agg = installments_payments()
    #ren = {col: "4IP_%s" % col for col in agg.columns}
    #agg.rename(columns=ren, inplace=True)
    #print("installments_payments df shape:", agg.shape)
    #df = df.join(agg, how='left', on='SK_ID_CURR')

    ##univariate_sample_analysis(df[list(agg.columns) + [PREDICTAND]], "../analysis/raw/installments_payments",
                               ##verbose=False,
                               ##predictand=PREDICTAND)

    #agg = credit_card_balance()
    #ren = {col: "5CCB_%s" % col for col in agg.columns}
    #agg.rename(columns=ren, inplace=True)
    #print("credit_card_balance df shape:", agg.shape)
    #df = df.join(agg, how='left', on='SK_ID_CURR')

    ##univariate_sample_analysis(df[list(agg.columns) + [PREDICTAND]], "../analysis/raw/credit_card_balance",
                               ##verbose=False,
                               ##predictand=PREDICTAND)

    columns_now = df.columns
    predictors.extend([c for c in columns_now if c not in columns_then])

    for col in df.columns:
        if df[col].dtype == "object":
            CATEGORICAL_FEATURES.append(col)


    #df_train_enc, encoders = encode_dataframe(select_sample(df, "train"), CATEGORICAL_FEATURES)
    #univariate_sample_analysis(df_train_enc, "../analysis/raw", verbose=False,
                               #predictand=PREDICTAND,
                               #categorical_features=CATEGORICAL_FEATURES,
                               #id_features=ID_FEATURES,
                               #encoders=encoders)

    # Show missing values
    missing = get_missing_values(df)
    print("\033[91;1mMissing values:\033[0;91m")
    print(missing)
    print("\033[0m")

    # Preprocessing

    # 1. Remove columns
    for col in ['FLAG_MOBIL', ]:
        df.drop(col, axis=1, inplace=True)
        predictors.remove(col)

    # 2. Replace some values

    # a) Flag doc (rare docs)
    doc_cols = [
        'FLAG_DOCUMENT_10',
        'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_17',
        'FLAG_DOCUMENT_2',
        'FLAG_DOCUMENT_4',
        'FLAG_DOCUMENT_7',
        ]
    df['FLAG_DOCUMENT_RARE'] = df[doc_cols].sum(axis=1)
    CATEGORICAL_FEATURES.append('FLAG_DOCUMENT_RARE')
    predictors.append('FLAG_DOCUMENT_RARE')

    df.drop(doc_cols, axis=1, inplace=True)
    for c in doc_cols:
        predictors.remove(c)

    # 3. Add column "is > 0"
    amt_cols = [col for col in df.columns if col.startswith('AMT_')]
    apt_cols = [c for c in df.columns if c.startswith('APARTMENTS_')]
    bsmt_cols = [c for c in df.columns if c.startswith('BASEMENTAREA_')]
    common_cols = [c for c in df.columns if c.startswith('COMMONAREA_')]
    def_cols = [c for c in df.columns if c.startswith('DEF_')]
    obs_cols = [c for c in df.columns if c.startswith('OBS_')]
    elev_cols = [c for c in df.columns if c.startswith('ELEVATORS_')]
    landarea_cols = [c for c in df.columns if c.startswith('LANDAREA_')]
    nonliv_cols = [c for c in df.columns if c.startswith('NONLIVING')]

    for col in amt_cols + apt_cols + bsmt_cols + common_cols + def_cols + elev_cols + landarea_cols + nonliv_cols + obs_cols + ["DAYS_LAST_PHONE_CHANGE"]:
        df[col + "_nz"] = (df[col] > 0).astype(int)
        CATEGORICAL_FEATURES.append(col + "_nz")
        predictors.append(col + "_nz")

    ## 4. Log-scale (TODO maybe this is not needed with QuantileTransformer ?)
    TO_LOG = {
        'AMT_ANNUITY': 1,
        'AMT_CREDIT': 1,
        'AMT_GOODS_PRICE': 1,
        'AMT_INCOME_TOTAL': 4,
        'AMT_REQ_CREDIT_BUREAU_MON': 1,
        'AMT_REQ_CREDIT_BUREAU_YEAR': 1,
        'APARTMENTS_AVG': 15,
        'APARTMENTS_MEDI': 15,
        'APARTMENTS_MODE': 15,
        "BASEMENTAREA_AVG": 15,
        "BASEMENTAREA_MEDI": 15,
        "BASEMENTAREA_MODE": 15,
        "COMMONAREA_AVG": 40,
        "COMMONAREA_MEDI": 40,
        "COMMONAREA_MODE": 40,
        "DEF_30_CNT_SOCIAL_CIRCLE": 40,
        "DEF_60_CNT_SOCIAL_CIRCLE": 40,
        "OBS_30_CNT_SOCIAL_CIRCLE": 40,
        "OBS_60_CNT_SOCIAL_CIRCLE": 40,
        "ELEVATORS_AVG": 6,
        "ELEVATORS_MEDI": 6,
        "ELEVATORS_MODE": 6,
        "ENTRANCES_AVG": 6,
        "ENTRANCES_MEDI": 6,
        "ENTRANCES_MODE": 6,
        "LANDAREA_AVG": 16,
        "LANDAREA_MEDI": 16,
        "LANDAREA_MODE": 16,
        "LIVINGAPARTMENTS_AVG": 16,
        "LIVINGAPARTMENTS_MEDI": 16,
        "LIVINGAPARTMENTS_MODE": 16,
        "LIVINGAREA_AVG": 16,
        "LIVINGAREA_MEDI": 16,
        "LIVINGAREA_MODE": 16,
        "NONLIVINGAPARTMENTS_AVG": 100,
        "NONLIVINGAPARTMENTS_MEDI": 100,
        "NONLIVINGAPARTMENTS_MODE": 100,
        "NONLIVINGAREA_AVG": 100,
        "NONLIVINGAREA_MEDI": 100,
        "NONLIVINGAREA_MODE": 100,
        'TOTALAREA_MODE': 1,
        }
    for col, log_lvl in TO_LOG.items():
        if col in df.columns:
            print('Taking log of column \033[94m%s, %i times\033[0m' % (col, log_lvl))
            while log_lvl > 0:
                df[col] = np.log1p(df[col])
                log_lvl -= 1
        else:
            print('\033[91mWarning column %s not in df\033[0m' % col)

    df['DAYS_EMPLOYED'][df['DAYS_EMPLOYED'] > 0] = np.nan
    df['DAYS_EMPLOYED_NAN'] = df['DAYS_EMPLOYED'].isnull()
    predictors.append('DAYS_EMPLOYED_NAN')
    CATEGORICAL_FEATURES.append('DAYS_EMPLOYED_NAN')

    ## re-plot preprocessed features before we encode with OneHotEncoder
    #df_train_enc, encoders = encode_dataframe(df_train, CATEGORICAL_FEATURES)
    #univariate_sample_analysis(df_train_enc, "../analysis/preprocessed", verbose=False,
                               #predictand=PREDICTAND,
                               #categorical_features=CATEGORICAL_FEATURES,
                               #id_features=ID_FEATURES,
                               #encoders=encoders)

    # Feature engineering
    print('Adding some other features')
    for i in range(3):
        for j in range(i, 3):
            i += 1
            j += 1
            df['EXT_SOURCE_%i%i' % (i, j)] = df['EXT_SOURCE_%i' % i] * df['EXT_SOURCE_%i' % j]


    # Encode categorical features, scale numerical ones
    print('Encoding features')
    to_remove = []
    to_add = []
    onehot_encoders = {}

    NUMERICAL_FEATURES = []

    for col in predictors:

        if col not in df.columns:
            print('\033[91mWarning column %s not in df\033[0m' % col)
            continue

        elif col in CATEGORICAL_FEATURES:
            print('LabelEncoder on %s' % col)
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            continue
            ## WARNING we use the fact that we can specify categorical features in LGBM
            #if df[col].nunique() <= 2:
                #print('LabelEncoder on %s' % col)
                #df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            #else:
                #print('OneHotEncoder on %s' % col)
                #lb = LabelEncoder().fit(df[col].astype(str))
                #values = lb.transform(df[col].astype(str)).reshape(-1, 1)

                #enc = OneHotEncoder().fit(values)
                #onehot_encoders[col] = enc
                #res = enc.transform(values)

                #for i, ax in enumerate(res.transpose()):
                    #ft_name = '%s_%s' % (col, lb.classes_[enc.active_features_[i]])
                    #ft_name = ft_name.replace('/', '-').replace('#', '-').replace('*', '-').replace(' ', '').replace('(', '-').replace(')', '-').replace('+', '-').replace('=', '-')
                    #arr = ax[0].toarray()
                    #df[ft_name] = ax.toarray().squeeze()
                    #to_add.append(ft_name)
                #to_remove.append(col)

        else:
            # Fill na
            # TODO explore correlations to fill with most frequent, median, etc...
            print('Scaling on %s' % col)
            null_values = df[col].isnull()
            if null_values.sum() > 0:

                # Skip nan column on aggregated features
                if not any([col.startswith('%i' % i) for i in range(1, 6)]):
                    print('-> Adding isNaN column')
                    df[col + '_nan'] = null_values.astype(int)
                    to_add.append(col + '_nan')
                    CATEGORICAL_FEATURES.append(col + '_nan')

            df[col].fillna(df[col].median(), inplace=True)

            if any([col.startswith('%i' % i) for i in range(1, 6)]):
                print('-> Using QuantileTransformer')
                df[col] = QuantileTransformer().fit_transform(
                df[col].values.reshape(-1, 1))

            else:
                print('-> Using RobustScaler')
                df[col] = RobustScaler().fit_transform(
                    df[col].values.reshape(-1, 1))
                #df[col] = QuantileTransformer().fit_transform(
                    #df[col].values.reshape(-1, 1))
                #df[col] = QuantileTransformer(output_distribution="normal").fit_transform(
                    #df[col].values.reshape(-1, 1))
            NUMERICAL_FEATURES.append(col)

            # We put the NaNs back
            df[col].iloc[np.where(null_values)[0]] = np.nan


    for col in to_remove:
        predictors.remove(col)
    predictors.extend(to_add)


    ## Plot transformed features
    #print('Plotting transformed features')
    #df_train_enc, encoders = encode_dataframe(select_sample(df, "train"), CATEGORICAL_FEATURES)
    #univariate_sample_analysis(df_train_enc, "../analysis/transformed", verbose=False,
                               #predictand=PREDICTAND,
                               #categorical_features=CATEGORICAL_FEATURES,
                               #id_features=ID_FEATURES,
                               #encoders=encoders)
    #del df_train_enc, encoders


    #print('Launching T-SNE')
    #fname = "../analysis/tsne/tsne.png"
    #os.makedirs(os.path.dirname(fname), exist_ok=True)

    #tsne_comps = tsne(df[NUMERICAL_FEATURES + ['TARGET']], fname, predictand="TARGET", binary=True)
    #for i, tsne_ax in enumerate(tsne_comps.transpose(), 1):
        #df['tsne%i' % i] = tsne_ax
        #predictors.append('tsne%i' % i)
    #tsne(df[CATEGORICAL_FEATURES], fname, predictand="TARGET", binary=True, visu_tsne=tsne_comps)


    df_train = select_sample(df, "train")
    df_test = select_sample(df, "test")

    categorical_features = [c for c in predictors if c in CATEGORICAL_FEATURES]

    feature_types = ['float'] * len(predictors)
    for i, c in enumerate(predictors):
        if c in categorical_features:
            feature_types[i] = 'int'


    X_train = df_train[predictors].values
    y_train = df_train[PREDICTAND].values.squeeze()
    X_test = df_test[predictors].values

    print('Shapes of X_train, X_test:')
    print(X_train.shape)
    print(X_test.shape)

    y_pred_xgb = []
    y_train_xgb = []
    y_pred_lgbm = []
    y_train_lgbm = []

    xgb_scores = []
    lgbm_scores = []

    xgb_params = {
        'silent': 1,
        'booster': 'gbtree',
        'eta': 0.02,
        'max_depth': 5,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 30}

    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 35,
        'max_depth': 5,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'verbose': -1,
        'num_boost_round': 15000,
        'early_stopping_rounds': 100,
        'save_binary': False,
        'nthread': 30}


    print('Launching K folds')

    nfolds = 10
    folds = StratifiedKFold(nfolds)

    xgb_feature_importances = []
    lgb_feature_importances_gain = []
    lgb_feature_importances_split = []

    for ifold, (trn_idx, val_idx) in enumerate(folds.split(
        X_train,
        y_train), 1):

        print("Fold nb. %i" % ifold)

        # XGBoost
        xgb_train = DMatrix(
            X_train[trn_idx, :],
            label=y_train[trn_idx],
            feature_names=predictors,
            feature_types=feature_types)

        xgb_val = DMatrix(
            X_train[val_idx, :],
            label=y_train[val_idx],
            feature_names=predictors,
            feature_types=feature_types)

        xgb_alltrain = DMatrix(
            X_train,
            label=y_train,
            feature_names=predictors,
            feature_types=feature_types)

        xgb_alltest = DMatrix(
            X_test,
            feature_names=predictors,
            feature_types=feature_types)

        print('\t[XGBoost ] training...')
        xgb_model = train_xgb(
            xgb_params,
            xgb_train,
            num_boost_round=15000,
            evals=[(xgb_val, 'valid_1')],
            verbose_eval=200,
            early_stopping_rounds=100,
            )

        y_pred = xgb_model.predict(X_train[val_idx, :])
        score = roc_auc_score(y_train[val_idx], y_pred)
        #print('\t[XGBoost ] best iteration: \033[92m%i\033[0m' % xgb_model.best_iteration)
        print('\t[XGBoost ] oof ROC-AUC is: \033[92m%.4f\033[0m' % score)

        y_train_xgb.append(xgb_model.predict(xgb_alltrain))
        y_pred_xgb.append(xgb_model.predict(xgb_alltest))
        xgb_scores.append(score)

        xgb_feature_importances.append(
            xgb_model.get_fscore())


        # Then LightGBM
        lgbm_train = Dataset(
            data=X_train[trn_idx, :],
            label=y_train[trn_idx],
            feature_name=predictors,
            categorical_feature=categorical_features)

        lgbm_val = Dataset(
            data=X_train[val_idx, :],
            label=y_train[val_idx],
            feature_name=predictors,
            categorical_feature=categorical_features)

        print('\t[LightGBM] training...')
        lgbm_model = train_lgb(
            lgbm_params,
            lgbm_train,
            num_boost_round=15000,
            early_stopping_rounds=100,
            valid_sets=[lgbm_train, lgbm_val],
            verbose_eval=200,
            categorical_feature=categorical_features,
            )

        y_pred = lgbm_model.predict(X_train[val_idx, :], num_iteration=lgbm_model.best_iteration)
        score = roc_auc_score(y_train[val_idx], y_pred)

        print('\t[LightGBM] best iteration: \033[92m%i\033[0m' % lgbm_model.best_iteration)
        print('\t[LightGBM] oof ROC-AUC is: \033[92m%.4f\033[0m' % score)

        y_train_lgbm.append(lgbm_model.predict(X_train, num_iteration=lgbm_model.best_iteration))
        y_pred_lgbm.append(lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration))
        lgbm_scores.append(score)

        lgb_feature_importances_gain.append(
            lgbm_model.feature_importance(importance_type="gain"))
        lgb_feature_importances_split.append(
            lgbm_model.feature_importance(importance_type="split"))


    y_pred_xgb = np.average(y_pred_xgb, axis=0, weights=np.array(xgb_scores) ** 2)
    y_pred_lgbm = np.average(y_pred_lgbm, axis=0, weights=np.array(lgbm_scores) ** 2)

    mean_xgb_scores = np.mean(xgb_scores)
    mean_lgbm_scores = np.mean(lgbm_scores)

    try:
        plot_feature_importances(predictors,
                                 xgb_feature_importances,
                                 "../classification/feature_importance/XGBoost",
                                 "fscore")
    except:
        import joblib
        with open('xgb_feature_importances.pyt', 'wb') as pyt:
            joblib.dump((predictors, xgb_feature_importances), pyt)
        print('Could not plot -- saved ft imp in xgb_feature_importances.pyt')
        print_exc()

    try:
        plot_feature_importances(predictors,
                                 lgb_feature_importances_gain,
                                 "../classification/feature_importance/LightGBM",
                                 "gain")
    except:
        import joblib
        with open('lgb_feature_importances_gain.pyt', 'wb') as pyt:
            joblib.dump((predictors, lgb_feature_importances_gain), pyt)
        print('Could not plot -- saved ft imp in lgb_feature_importances_gain.pyt')
        print_exc()

    try:
        plot_feature_importances(predictors,
                                 lgb_feature_importances_split,
                                 "../classification/feature_importance/LightGBM",
                                 "split")
    except:
        import joblib
        with open('lgb_feature_importances_split.pyt', 'wb') as pyt:
            joblib.dump((predictors, lgb_feature_importances_split), pyt)
        print('Could not plot -- saved ft imp in lgb_feature_importances_split.pyt')
        print_exc()


    fname = 'LGB.csv'
    df_lgb = pd.DataFrame({
        "SK_ID_CURR": df_test.index,
        "TARGET": y_pred_lgbm})
    df_lgb.to_csv(fname, index=False, sep=',')
    print('LightGBM predictions saved to \033[92m%s\033[0m' % fname)

    fname = 'XGB.csv'
    df_xgb = pd.DataFrame({
        "SK_ID_CURR": df_test["SK_ID_CURR"],
        "TARGET": y_pred_xgb})
    df_xgb.to_csv(fname, index=False, sep=',')
    print('XGBoost predictions saved to \033[92m%s\033[0m' % fname)

    hybrid_pred = np.average(
        [y_pred_lgbm, y_pred_xgb],
        axis=0,
        weights=[mean_lgbm_scores, mean_xgb_scores])
    fname = 'Hybrid.csv'
    df_both = pd.DataFrame({
        "SK_ID_CURR": df_test["SK_ID_CURR"],
        "TARGET": hybrid_pred})
    df_both.to_csv(fname, index=False, sep=',')
    print('Hybrid predictions saved to \033[92m%s\033[0m' % fname)
