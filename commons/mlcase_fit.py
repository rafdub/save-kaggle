#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Read, clean, engineer data, and train models.
Use in script mode with -h for help.
'''

import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, Binarizer
import category_encoders as ce

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from sklearn.tree import export_graphviz

import datetime

# joblib for fast file dump & load
import joblib


from analysis import univariate_sample_analysis, tsne, plot_pca, DATE_ZERO


# Directories & files
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, 'ml_case_data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
ENCODERS_FILE = os.path.join(MODELS_DIR, 'encoders.pyt')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pyt')

ANALYSIS_DIR = os.path.join(ROOT_DIR, 'analysis')
UNIVARIATE_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, 'univariate')
BIVARIATE_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, 'bivariate')
MULTIVARIATE_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, 'multivariate')


# Name of the predictand (= target feature)
PREDICTAND = 'churn'

# Types of features
CATEGORICAL_FEATURES = [
    'activity_new',
    'campaign_disc_ele',
    'channel_sales',
    'origin_up',
    ]

DATE_FEATURES = [
    'date_activ',
    'date_end',
    'date_first_activ',
    'date_modif_prod',
    'date_renewal',
    ]

BOOL_COLS = [
    'has_gas',
    ]


# Columns to remove during data tidying
REMOVE_COLUMNS_CLEANING = [
    'campaign_disc_ele',
    ]

# Columns to remove from predictors
# remove the ones that are surely not linked to the churn rate
REMOVE_FROM_PREDICTOR = [
    'campaign_disc_ele',
    'missing_campaign_disc_ele',

    'cons_gas_12m',
    'cons_last_month',

    'forecast_base_bill_ele',
    'forecast_base_bill_year',
    'forecast_bill_12m',
    'forecast_cons',
    'forecast_discount_energy',

    'imp_cons',
    ]


# Feature engineering

# Right-skewed variables -> use sqrt (or log)
# (ml algorithms prefer normally-distributed features)
RIGHT_SKEWED = [
    'forecast_meter_rent_12m',
    'imp_cons',
    'pow_max',
    'cons_12m',
    'cons_gas_12m',
    'cons_last_month',
    'margin_gross_pow_ele',
    'margin_net_pow_ele',
    'net_margin',
    'pow_max',
    'forecast_cons',
    #'forecast_discount_energy',
    'forecast_bill_12m',
    'forecast_base_bill_year',
    'forecast_base_bill_ele',
    'forecast_cons_12m',
    'forecast_cons_year',
    'forecast_cons',
    ]

# For left-skewed variable we take the log(1-x*)
LEFT_SKEWED = [
    'date_renewal',
    'date_modif_prod',
    'date_first_activ',
    'date_activ',
    ]

# Configuration of some specific predictors we want
# by default we compute all the time periods between date features
# these are probably more insightful than actual dates
PREDICTORS = {
    'end-renewal': ('get_period', ('date_renewal', 'date_end'), {}),
    'end-activ': ('get_period', ('date_activ', 'date_end'), {}),
    'end-modif_prod': ('get_period', ('date_modif_prod', 'date_end'), {}),
    'renewal-activ': ('get_period', ('date_activ', 'date_renewal'), {}),
    'renewal-first_activ': ('get_period', ('date_first_activ', 'date_renewal'), {}),
    'renewal-modif_prod': ('get_period', ('date_modif_prod', 'date_renewal'), {}),
    'modif_prod-activ': ('get_period', ('date_activ', 'date_modif_prod'), {}),
    'modif_prod-first_activ': ('get_period', ('date_first_activ', 'date_modif_prod'), {}),
    }
PREDICTORS.update({name: ('get_time', (name,), {}) for name in DATE_FEATURES})
#PREDICTORS.update({'fixed_cost_pow_ele': ('substract', ('margin_gross_pow_ele', 'margin_net_pow_ele'), {})})

# Remove null entries along these columns
REMOVE_ON_MISSING = [
    'date_end',
    'margin_gross_pow_ele',
    'margin_net_pow_ele',
    'net_margin',
    'pow_max',
    ]


class SortedLabelEncoder:

    '''Like a LabelEncoder but which sorts categories
    depending on the predictand mean value
    '''

    def __init__(self):
        self.unique_sorted = None

    def fit(self, df, column, target=PREDICTAND):
        unique = df[column].unique()
        means = df.groupby(column)[target].mean()
        self.unique_sorted = list(means.sort_values().index)

    def transform(self, series):
        return series.apply(lambda x:
                                self.unique_sorted.index(x)
                                if x in self.unique_sorted else -1)



class ClientDataFrame(pd.DataFrame):

    '''
    pd.DataFrame with utilities to compute predictors using instance methods
    '''

    @classmethod
    def read_csv(cls, file_name):
        df = pd.read_csv(file_name, parse_dates=DATE_FEATURES, infer_datetime_format=True)
        df = df.set_index('id')

        for col in [c for c in BOOL_COLS if c in df.columns]:
            df[col] = df[col] == 't'

        return ClientDataFrame(df)


    def get_period(self, from_, to_):
        '''New column: the number of days between 2 date columns'''
        col = '%s-%s' % (from_, to_)
        self[col] = self[to_] - self[from_]
        self[col] = self[col].apply(lambda x: x.total_seconds() / 86400.)
        return self[col]

    def get_time(self, col):
        '''To convert dates to numeric features,
        we transform these to "days since DATE_ZERO" (can be negative)
        '''
        return (self[col] - DATE_ZERO).apply(lambda x: x.total_seconds() / 86400.)

    def substract(self, to_, from_):
        return self[to_] - self[from_]


def read_price_data(file_name):
    df = pd.read_csv(file_name, parse_dates=['price_date'], infer_datetime_format=True)
    df = df.set_index(['id', 'price_date'])
    return df


def load_sample(kind='training', force=False):
    '''Reads a whole sample, training or test'''
    dumpfile = '%s.pyt' % kind
    if not force and os.path.isfile(dumpfile):
        try:
            df = joblib.load(dumpfile)
            return df
        except:
            pass

    client_file = os.path.join(DATA_DIR, 'ml_case_%s_data.csv' % kind)
    price_file = os.path.join(DATA_DIR, 'ml_case_%s_hist_data.csv' % kind)

    # Read the client data
    df = ClientDataFrame.read_csv(client_file)
    df.sort_index()

    # Read the price data
    dfp = read_price_data(price_file)

    # Compute the mean (fix/var) prices
    # and the first/last prices
    mean_prices = dfp.groupby('id').mean()
    for column in mean_prices.columns:
        df['mean_%s' % column] = mean_prices[column]
    # We could add more features for further investigation !

    # If this is the training sample we read the churn flag
    if kind == 'training':
        churn_file = os.path.join(DATA_DIR, 'ml_case_%s_output.csv' % kind)
        dfc = pd.read_csv(churn_file)

        # Make sure they correspond to the right clients
        dfc = dfc.set_index(['id'])
        df[PREDICTAND] = dfc['churn']

    with open(dumpfile, 'wb') as pyt:
        print('Saving sample to %s' % pyt)
        joblib.dump(df, pyt)

    return df


def clean_sample(df, verbose=0):
    '''Cleans the input DataFrame based on the preliminary analyses.

    Handles missing values (use replacement by median, other feature or 0),
    removes some columns.
    '''

    # date_modif_prod, date_renewal, date_first_activ
    # -> if NaN, we assume it is the date_activ
    for col in ['date_modif_prod', 'date_renewal', 'date_first_activ']:
        inds = pd.isnull(df[col])
        if verbose > 2:
            print('\tFilling %i values of %s using date_activ' % (np.sum(inds), col))
        df[col][inds] = df['date_activ'][inds]

    # Flag some entries as invalid if they are null
    inds = np.ones_like(df.index, dtype=bool)
    for col in REMOVE_ON_MISSING:
        inds &= np.logical_not(pd.isnull(df[col]))

    # date_end cannot be prior to 1st Jan. 2016 as these are Jan. 2016 clients
    inds &= (df['date_end'] >= datetime.datetime(2016, 1, 1))

    # Fill missing values on numerical features - infer median
    for col in df.columns:
        if col not in CATEGORICAL_FEATURES + DATE_FEATURES:
            if verbose > 2:
                print('\tFilling %i values along column %s' % (np.sum(pd.isnull(df[col])), col))

            if np.sum(pd.isnull(df[col])):
                # Fill the values using the median
                # NB: In this case, flagging missing values didn't improve accuracy
                df[col] = df[col].fillna(df[col].median())

    # Fill missing values on all categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if np.sum(pd.isnull(df[col])):
                # Add a column to flag the missing values
                df['missing_%s' % col] = pd.isnull(df[col])
                df[col] = df[col].fillna('unknown')

    # Remove some features
    df = df[[col for col in df.columns if col not in REMOVE_COLUMNS_CLEANING]]

    # Remove samples flagged as invalid during the cleaning process
    if verbose > 1:
        print('\tRemoving %i null values of %i' % (np.sum(np.logical_not(inds)), len(inds)))
    df = df.loc[df.index[inds], :]

    return ClientDataFrame(df)


def get_predictors_and_encoders(df, feature_engineering=True, verbose=0, encoders=None):
    '''
    df: input DataFrame (client + price data)
    feature_engineering: whether or not to perform it

    Returns
        a DataFrame containing the predictors (and the predictand), and
        a dictionary containing the encoders for each category

    The way most of the predictors are built is specified in the PREDICTORS variable
    '''
    data = {}
    for key in df.columns:
        if key in REMOVE_FROM_PREDICTOR: continue
        if not key.startswith('date_'):
            data[key] = df[key]

    for key, (method, args, kwargs) in PREDICTORS.items():
        data[key] = getattr(df, method)(*args, **kwargs)

    pred = pd.DataFrame(data, index=df.index)
    pred.index.name = df.index.name

    if feature_engineering:

        if verbose:
            print('Performing feature engineering on the predictors')

        # Convert features with a right-skewed distribution
        # so they are "more normally" distributed.
        # Take sqrt or log, depending on the observations.
        for col in pred.columns:
            if col in RIGHT_SKEWED:
                if verbose > 1:
                    print('\tTaking the logarithm of right-skewed feature %s' % col)
                pred[col][pred[col] < 0] = 0
                pred[col] = np.log(pred[col] + 1)

            elif col in LEFT_SKEWED:
                if verbose > 1:
                    print('\tTaking the logarithm of the opposite of left-skewed feature %s' % col)
                pred[col] = np.log(pred[col].max() - pred[col] + 1)

    if encoders is None:
        encoders = {}

    # Finally, use label encoders for categorical features
    for col in CATEGORICAL_FEATURES:
        if col not in pred.columns: continue

        encoder = encoders.get(col, None)
        if encoder is None:
            encoder = SortedLabelEncoder()
            # Other ways of encoding were tested (i.e. Helmert, one-hot)
            # this one yielded the best results
            encoder.fit(pred, col, PREDICTAND)

        pred[col] = encoder.transform(pred[col])
        encoders[col] = encoder

    return pred.dropna(), encoders



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='''Read, clean, engineer the data
        Then train models and save them''',
        )

    parser.add_argument(
        '--force',
        action='store_true',
        help='''Force the algorithm to use the .csv files
        instead of previously loaded data''')

    parser.add_argument(
        '--groupby',
        type=str,
        help='''Feature(s) on which to focus the analysis,
        will use the category with the highest churn''',
        default=None)

    parser.add_argument(
        '--random_state',
        type=int,
        help='Set a random state for regressions & classifiers',
        default=28)

    parser.add_argument(
        '--n_estimators',
        type=int,
        help='Number of estimators for ensemble regressions & classifiers',
        default=1000)

    parser.add_argument(
        '--max_depth',
        type=int,
        help='Max depth for decition trees & random forests',
        default=15)

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Realize basic analyses to visualize the data')

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save the results')

    # Options to control t-SNE visualization
    parser.add_argument(
        '--tsne',
        action='store_true',
        help='Realize a t-SNE graph of the predictors')
    parser.add_argument(
        '-n', '--number',
        type=int,
        help='Number of samples for the TSNE (impacts generation time & CPU usage)',
        default=1000)
    parser.add_argument(
        '-p', '--perplexity',
        type=float,
        help='perplexity for TSNE',
        default=30)

    parser.add_argument(
        '--fit_on_pca',
        action='store_true',
        help='Use PCs instead of original (cleaned & engineered) features as model input',
        )

    #parser.add_argument(
        #'--load_previous',
        #action='store_true',
        #help='Do not learn again, use previous algorithms',
        #)

    parser.add_argument(
        '-g', '--debug',
        action='store_true')

    parser.add_argument(
        '-v', '--verbose',
        help='Verbosity level (count) - from 0 (not verbose) to 3 (very verbose)',
        action='count')

    args = parser.parse_args()

    if args.verbose is None:
        args.verbose = 0

    if args.groupby is not None:
        assert(args.groupby in CATEGORICAL_FEATURES), "Must be a category"

    # Read the data
    if args.verbose: print('Loading the sample...')

    df = load_sample('training', force=args.force)
    if args.verbose > 1:
        print('\033[1mLoaded the sample\033[0m')
        print('\n\t'.join(df.columns))
        print(df.head())

    if args.verbose: print('Cleaning the sample...')
    if os.path.isfile('cleaned.pyt'):
        df = joblib.load('cleaned.pyt')
    else:
        df = clean_sample(df, verbose=args.verbose)
        with open('cleaned.pyt', 'wb') as pyt:
            joblib.dump(df, pyt)

    if args.analyze:

        # Make some figures to visualize the cleaned data
        df_for_analysis = df.copy()
        for col in df_for_analysis.columns:

            # Categorical features are randomly encoded for visualization
            if col in CATEGORICAL_FEATURES:
                df_for_analysis[col] = LabelEncoder().fit_transform(df_for_analysis[col])

            # Date features are converted to numerical values
            elif col in DATE_FEATURES:
                df_for_analysis[col] = df.get_time(col)


        univariate_sample_analysis(
            df_for_analysis,
            UNIVARIATE_ANALYSIS_DIR + '_cleaned',
            categorical_features=CATEGORICAL_FEATURES+BOOL_COLS,
            verbose=args.verbose)

        # Plot a PCA
        columns_for_pca = [c for c in df_for_analysis.columns if c not in CATEGORICAL_FEATURES]
        plot_pca(df_for_analysis[columns_for_pca], os.path.join(ROOT_DIR, 'pca.png'), npc=7)


    if args.tsne:
        # t-SNE visualization is a good way to visualize
        # high-dimensional datasets - it can identify groups
        columns_for_tsne = [
            c for c in df.columns if c not in CATEGORICAL_FEATURES + DATE_FEATURES]

        tsne(
            df[columns_for_tsne],
            os.path.join(ROOT_DIR, 'TSNE', 'cleaned_data', 'tsne.png'),
            nb=args.number,
            title='Cleaned DataFrame',
            perplexity=args.perplexity,
            )

    if args.verbose: print('Getting the predictors...')
    predictors, encoders = get_predictors_and_encoders(df, feature_engineering=True)

    # Save the encoders for later use
    if not args.analyze:
        with open(ENCODERS_FILE, 'wb') as pyt:
            joblib.dump(encoders, pyt)
        if args.verbose:
            print('Categorical features encoders saved to \033[92m%s\033[0m' % ENCODERS_FILE)


    if args.analyze:
        # Visualize the engineered data to make sure it is correctly distributed
        univariate_sample_analysis(
            predictors,
            UNIVARIATE_ANALYSIS_DIR + '_engineered',
            categorical_features=CATEGORICAL_FEATURES+BOOL_COLS,
            verbose=args.verbose)
        pca = plot_pca(predictors, os.path.join(ANALYSIS_DIR, 'pca_engineered.png'), npc=7)

        if args.tsne:
            # t-SNE visualization is a good way to visualize
            # high-dimensional datasets - it can identify groups
            #columns_for_tsne = [
                #c for c in predictors.columns if c not in CATEGORICAL_FEATURES + DATE_FEATURES]
            columns_for_tsne = [c for c in predictors.columns if 'price' in c and 'forecast' not in c]

            print('Fitting...', columns_for_tsne)
            tsne_coords = tsne(
                predictors[columns_for_tsne],
                os.path.join(ROOT_DIR, 'TSNE', 'engineered_data', 'tsne.png'),
                nb=args.number,
                title='Engineered DataFrame',
                perplexity=55,
                )
            print('Plotting...')
            tsne(
                predictors,
                os.path.join(ROOT_DIR, 'TSNE', 'engineered_data', 'tsne.png'),
                nb=args.number,
                title='Engineered DataFrame',
                perplexity=55,
                visu_tsne=tsne_coords,
                )

        # We stop here in analysis mode
        sys.exit(1)

    #pca = plot_pca(predictors, os.path.join(ANALYSIS_DIR, 'pca_engineered.png'), npc=7)
    #print(pca.shape)
    #predictors['pca_6'] = pca[:, 5]

    # Now that we cleaned & engineered our data we can move on to modeling
    predictors = predictors[[c for c in predictors.columns if c not in REMOVE_COLUMNS_CLEANING]]

    # Do we want to focus on a specific category ?
    if args.groupby is not None:
        if args.verbose:
            print("Select only the %s category with the highest churn rate" % args.groupby)

        # We select the category of the provided feature with the highest churn
        category = predictors[args.groupby].unique()
        means = predictors.groupby(args.groupby)[PREDICTAND].mean()
        counts = predictors.groupby(args.groupby)[PREDICTAND].count()
        if args.verbose > 1:
            for o, m, c in zip(category, means, counts):
                print('Category %s: mean churn of %.3f, %i values' % (o, m, c))

        # Discard categories with less than 10% of the customers
        counts /= np.sum(counts)
        means[counts < 0.1] = 0
        i = np.argmax(means)

        if args.verbose > 1:
            print("The %i-th %s (%s) has the highest churn rate" % (
                i, args.groupby, category[i]))

        # Update the DataFrame of predictors
        predictors = predictors.groupby(args.groupby).get_group(i)
        predictors = predictors[[c for c in predictors.columns if c != args.groupby]]


    if args.verbose > 1:
        print('\033[1mGot the predictors\033[0m')
        print('\n\t'.join(predictors.columns))
        if args.verbose > 2: print(predictors)


    # For t-SNE visualization of the model output,
    # we only select the price features
    # to try to highlight a correlation between prices
    # and churn rate
    COLS_FOR_TSNE = [c for c in predictors.columns
                         if 'price' in c] + [PREDICTAND]

    # Get the list of real predictors
    # (the "predictors" array also contains the predictand)
    predictor_names = [c for c in predictors.columns if c != PREDICTAND]


    # If we are running in learn mode, fit and save the Helmert encoder
    # based on the whole sample
    columns_to_encode = [col for col in CATEGORICAL_FEATURES if col in predictor_names]

    if args.verbose > 0:
        print('Regressions & Classifications based on the columns:')
        print('\033[93m%s\033[0m' % predictor_names)


    # Split train and test in our overall train sample because
    # we want to assess the model accuracy (without overfitting)
    X_train, X_test, y_train, y_test = train_test_split(
        predictors[predictor_names].values,
        predictors[PREDICTAND].values,
        test_size=0.3,
        )


    if args.verbose > 1:
        print('Training sample size', X_train.shape)
        print('Test sample size', X_test.shape)


    # In the following we test numerous machine learning models:
    # Decision Trees, Random Forests, Gradient Boosting (ensemble methods)
    # and Multilayer Perceptron (basic artificial neural network)

    # We print accuracy report using various metrics:
    # classification_report provides:
    # - prediction = if we flagged "True", % of time it is indeed True
    # - recall = % of all the actual True we flagged as such
    # - f1-score = combination of the two
    # also add the requires area under ROC curve and Brier loss

    # We also plot t-SNE graphs to visualize the actual churn and the predicted one

    # We also save the resulting fitted models for later use
    # (we fit them again on the whole training data)

    # This part could have been cleaner / shorter...
    os.makedirs(MODELS_DIR, exist_ok=True)

    tsne_dir = os.path.join(ROOT_DIR, 'tsne', 'training')
    tsne_coords = None

    def launch_tsne(base_file, title, X_test, y_test, y_predict, predictor_names, tsne_coords, cmap='viridis'):
        '''t-SNE visualization of the results'''
        df_for_tsne = pd.DataFrame(
            np.concatenate([
                X_test,
                np.expand_dims(y_test, axis=1),
                np.expand_dims(y_predict, axis=1)
                ], axis=1),
            columns=predictor_names + [PREDICTAND] + ['predicted'])

        return tsne(
            df_for_tsne,
            base_file,
            nb=15000,
            title=title,
            perplexity=args.perplexity,
            visu_tsne=tsne_coords,
            cmap=cmap,
            )

    def print_regressor_report(name, X_train, y_train, X_test, y_test, y_predict, color_code="92", threshold=0.5):
        '''Shows metrics to assess a regression model accuracy
        also perform these on the training sample to assess the tendancy to over-fitting
        '''
        print('\033[%sm' % color_code)
        print('====================================')
        print('%s Regression report' % name)
        #print('\tPredicted churn probability varies from %.3f to %.3f' % (y_predict.min(), y_predict.max()))
        y_predict_n1 = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        y_predict = np.maximum(0, np.minimum(1, y_predict))
        print('\tPredicted a churn rate of %.2f vs. true %.2f' % (
            np.mean(y_predict > threshold), np.mean(y_test)))

        for threshold in [0.5, 0.7, 0.8, 0.9]:
            print('\tthreshold: %s' % threshold)
            print(classification_report(y_test, y_predict > threshold))
            print('\tand on the training sample')
            print(classification_report(y_train, reg.predict(X_train) > threshold))

        print('====================================')
        print('| (N1) Area under ROC curve: %.4f |' % roc_auc_score(y_test, y_predict_n1))
        print('| (N1) Brier loss score: %.4f |' % brier_score_loss(y_test, y_predict_n1))
        print('| (N2) Area under ROC curve: %.4f |' % roc_auc_score(y_test, y_predict))
        print('| (N2) Brier loss score: %.4f |' % brier_score_loss(y_test, y_predict))
        print('====================================')
        print('\033[0m')

    def print_classifier_report(name, X_train, y_train, X_test, y_test, y_predict, color_code="32"):
        '''Shows metrics to assess a regression model accuracy
        also perform these on the training sample to assess the tendancy to over-fitting
        '''
        print('\033[%sm' % color_code)
        print('====================================')
        print('%s Classifier report' % name)
        print('\tPredicted a churn rate of %.2f vs. true %.2f' % (
            np.mean(y_predict), np.mean(y_test)))

        print(classification_report(y_test, y_predict))
        print('\tand on the training sample')
        print(classification_report(y_train, clf.predict(X_train)))

        print('====================================')
        print('| Area under ROC curve: %.4f |' % roc_auc_score(y_test, y_predict))
        print('| Brier loss score: %.4f |' % brier_score_loss(y_test, y_predict))
        print('====================================')
        print('\033[0m')

    def get_roc(y_test, y_predict):
        # only for regressors !
        thresholds = np.linspace(0.5, 1, 501)
        tpr = np.array([np.sum(np.logical_and(y_predict >= t, y_test)) / np.sum(y_test) for t in thresholds])
        tnr = np.array([np.sum(np.logical_and(y_predict < t, 1-y_test)) / np.sum(1-y_test) for t in thresholds])
        return tnr, tpr


    def save_model(model, filename):
        if args.save:
            model.fit(predictors[predictor_names], predictors[PREDICTAND])
            with open(os.path.join(MODELS_DIR, filename), 'wb') as dump:
                joblib.dump(model, dump)


    reg = GradientBoostingRegressor(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        )
    reg.fit(X_train, y_train)
    y_predict = reg.predict(X_test)

    if args.verbose:
        print_regressor_report(
            "Gradient Boosting",
            X_train, y_train,
            X_test, y_test, y_predict,
            color_code="92")

    if args.tsne:
        tsne_coords = launch_tsne(
            os.path.join(tsne_dir, 'gradbr', 'tsne_gradbr.png'),
            'Gradient Boosting Regressor',
            X_test, y_test, y_predict,
            predictor_names,
            None,
            cmap='magma',
            )

    save_model(reg, 'gradient_boosting_regressor.pyt')
    roc_gradient_boosting = get_roc(y_test, y_predict)


    #clf = GradientBoostingClassifier(
        #random_state=args.random_state,
        #n_estimators=args.n_estimators,
        #)
    #clf.fit(X_train, y_train)
    #y_predict = clf.predict(X_test)

    #if args.verbose:
        #print_classifier_report(
            #"Gradient Boosting",
            #X_train, y_train,
            #X_test, y_test, y_predict,
            #color_code="32")

    #if args.tsne:
        #launch_tsne(
            #os.path.join(tsne_dir, 'gradbc', 'tsne_gradbc.png'),
            #'Gradient Boosting Classifier',
            #X_test, y_test, y_predict,
            #predictor_names,
            #tsne_coords,
            #cmap='viridis',
            #)

    #save_model(clf, 'gradient_boosting_classifier.pyt')


    # scaler + neural network dimensions
    ss = StandardScaler().fit(X_train)

    # Save the standard scaler fitted on the whole training data
    ss_whole = StandardScaler().fit(predictors[predictor_names])
    with open(SCALER_FILE, 'wb') as pyt:
        joblib.dump(ss_whole, pyt)

    hidden_layer_sizes = [60, 60, 60, 80, 60, 60, 60]

    reg = MLPRegressor(
        random_state=args.random_state,
        hidden_layer_sizes=hidden_layer_sizes)

    reg.fit(ss.transform(X_train), y_train)
    y_predict = reg.predict(ss.transform(X_test))

    if args.verbose:
        print_regressor_report(
            "Multilayer Perceptron",
            X_train, y_train,
            X_test, y_test, y_predict,
            color_code="94")

    if args.tsne:
        launch_tsne(
            os.path.join(tsne_dir, 'mlpr', 'tsne_mlpr.png'),
            'Multilayer Perceptron Regressor',
            X_test, y_test, y_predict,
            predictor_names,
            tsne_coords,
            cmap='magma',
            )

    save_model(reg, 'mlp_regressor.pyt')
    roc_mlp = get_roc(y_test, y_predict)


    #clf = MLPClassifier(
        #random_state=args.random_state,
        #hidden_layer_sizes=hidden_layer_sizes,
        #)
    #clf.fit(ss.transform(X_train), y_train)
    #y_predict = clf.predict(ss.transform(X_test))

    #if args.verbose:
        #print_classifier_report(
            #"Multilayer Perceptron",
            #X_train, y_train,
            #X_test, y_test, y_predict,
            #color_code="34")

    #if args.tsne:
        #launch_tsne(
            #os.path.join(tsne_dir, 'mlpc', 'tsne_mlpc.png'),
            #'Neural Network Classifier',
            #X_test, y_test, y_predict,
            #predictor_names,
            #tsne_coords,
            #cmap='viridis',
            #)

    #save_model(clf, 'mlp_classifier.pyt')


    # Decision trees
    reg = DecisionTreeRegressor(
        random_state=args.random_state,
        max_depth=args.max_depth,
        )
    reg.fit(X_train, y_train)
    y_predict = reg.predict(X_test)

    if args.verbose:
        print_regressor_report(
            "Decision Tree",
            X_train, y_train,
            X_test, y_test, y_predict,
            color_code="93")

    if args.tsne:
        tsne_coords = launch_tsne(
            os.path.join(tsne_dir, 'dtreer', 'tsne_dtreer.png'),
            'Decision Tree Regressor',
            X_test, y_test, y_predict,
            predictor_names,
            tsne_coords,
            cmap='magma',
            )

    save_model(reg, 'decision_tree_regressor.pyt')
    roc_decision_tree = get_roc(y_test, y_predict)


    #clf = DecisionTreeClassifier(
        #random_state=args.random_state,
        #max_depth=args.max_depth,
        #)
    #clf.fit(X_train, y_train)
    #y_predict = clf.predict(X_test)

    #if args.verbose:
        #print_classifier_report(
            #"Decision Tree",
            #X_train, y_train,
            #X_test, y_test, y_predict,
            #color_code="33")

    ## In case of a decision tree we can also output a tree graph
    #export_graphviz(clf,
                    #out_file=os.path.join(ROOT_DIR, "decision_tree.dot"),
                    #feature_names=predictor_names)

    #if args.tsne:
        #launch_tsne(
            #os.path.join(tsne_dir, 'dtreec', 'tsne_dtreec.png'),
            #'Decision Tree Classifier',
            #X_test, y_test, y_predict,
            #predictor_names,
            #tsne_coords,
            #)

    #save_model(clf, 'decision_tree_classifier.pyt')


    ## Random forest
    #clf = RandomForestClassifier(
        #random_state=args.random_state,
        #n_estimators=args.n_estimators,
        #max_depth=args.max_depth)
    #clf.fit(X_train, y_train)
    #y_predict = clf.predict(X_test)

    #if args.verbose:
        #print_classifier_report(
            #"Random Forest",
            #X_train, y_train,
            #X_test, y_test, y_predict,
            #color_code="35")

    #if args.tsne:
        #launch_tsne(
            #os.path.join(tsne_dir, 'randfc', 'tsne_randfc.png'),
            #'Random Forest Classifier',
            #X_test, y_test, y_predict,
            #predictor_names,
            #tsne_coords,
            #cmap='magma',
            #)

    #save_model(clf, 'random_forest_classifier.pyt')


    reg = RandomForestRegressor(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth)
    reg.fit(X_train, y_train)
    y_predict = reg.predict(X_test)

    if args.verbose:
        print_regressor_report(
            "Random Forest",
            X_train, y_train,
            X_test, y_test, y_predict,
            color_code="95")

    if args.tsne:
        launch_tsne(
            os.path.join(tsne_dir, 'randfr', 'tsne_randfr.png'),
            'Random Forest Regressor',
            X_test, y_test, y_predict,
            predictor_names,
            tsne_coords,
            cmap='magma',
            )

    save_model(reg, 'random_forest_regressor.pyt')
    roc_random_forest = get_roc(y_test, y_predict)

    import matplotlib.pyplot as plt
    import stats.visualization.colors as svc
    plt.figure(figsize=(8, 8))
    for i, curve in enumerate([roc_gradient_boosting, roc_mlp, roc_decision_tree, roc_random_forest]):
        plt.plot(1-curve[0], curve[1], lw=2, color=svc.COLOR_HUE[i])
    plt.grid()
    plt.legend(['Gradient boosting', 'Neural network', 'Decision tree', 'Random forest'], loc='lower right')
    #plt.plot([0, 1], [0, 1], 'k-')
    #for i, curve in enumerate([roc_gradient_boosting, roc_mlp, roc_decision_tree, roc_random_forest]):
        #plt.plot(1-curve[0][250], curve[1][250], lw=2, marker='o', ms=8, color=svc.COLOR_HUE[i])
    plt.xlabel('False negative rate')
    plt.ylabel('True positive rate')
    plt.savefig('roc_curves_zoom0.5.png')
    plt.close()

