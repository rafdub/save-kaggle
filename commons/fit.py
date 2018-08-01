import os
import joblib

import numpy as np
import pandas as pd

from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, PassiveAggressiveRegressor, Lasso, Ridge
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor

from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss, mean_squared_log_error, r2_score, explained_variance_score, mean_squared_error

from scipy.optimize import curve_fit

#from yellowbrick.regressor import PredictionError, ResidualsPlot
import matplotlib.pyplot as plt


REGRESSORS = OrderedDict()
#REGRESSORS['Lasso'] = Lasso
REGRESSORS['XGBoost'] = XGBRegressor
REGRESSORS['LightGBM'] = LGBMRegressor
#REGRESSORS['MLP'] = MLPRegressor
#REGRESSORS['DecisionTree'] = DecisionTreeRegressor
REGRESSORS['RandomForest'] = RandomForestRegressor
REGRESSORS['GradientBoosting'] = GradientBoostingRegressor
#REGRESSORS['LogisticRegression'] = LogisticRegression
REGRESSORS['AdaBoost'] = AdaBoostRegressor
#REGRESSORS['Lasso'] = Lasso
#REGRESSORS['Ridge'] = Ridge
#REGRESSORS['PassiveAgressive'] = PassiveAggressiveRegressor


CLASSIFIERS = OrderedDict()
#CLASSIFIERS['XGBoost'] = XGBClassifier
CLASSIFIERS['MLP'] = MLPClassifier
#CLASSIFIERS['DecisionTree'] = DecisionTreeClassifier
CLASSIFIERS['RandomForest'] = RandomForestClassifier
CLASSIFIERS['GradientBoosting'] = GradientBoostingClassifier
CLASSIFIERS['AdaBoost'] = AdaBoostClassifier


def linreg(x, a, b):
    return a * x + b


def print_regressor_report(name, X_train, y_train, X_test, y_test, y_predict, color_code="92", threshold=0.5, predictand_name="predictand"):
    '''Shows metrics to assess a regression model accuracy
    also perform these on the training sample to assess the tendancy to over-fitting
    '''
    print('\033[%sm' % color_code)
    print('====================================')
    print('%s Regression report' % name)
    print('\tPredicted %s varies from %.3f to %.3f' % (predictand_name, y_predict.min(), y_predict.max()))
    print('| Corr:               {:8.4f}'.format(np.corrcoef(y_predict, y_test)[0, 1]))
    print('| MSE:                {:8.4f}'.format(mean_squared_error(y_test, y_predict)))
    print('| Explained variance: {:8.4f}'.format(explained_variance_score(y_test, y_predict)))
    print('| R2:                 {:8.4f}'.format(r2_score(y_test, y_predict)))
    print('====================================')
    print('\033[0m')

def print_classifier_report(name, X_train, y_train, X_test, y_test, y_predict, color_code="32", predictand_name="predictand"):
    '''Shows metrics to assess a regression model accuracy
    also perform these on the training sample to assess the tendancy to over-fitting
    '''
    print('\033[%sm' % color_code)
    print('====================================')
    print('%s Classifier report' % name)
    print('\tPredicted a %s rate of %.2f vs. true %.2f' % (
        predictand_name, np.mean(y_predict), np.mean(y_test)))

    print(classification_report(y_test, y_predict))
    #print('\tand on the training sample')
    #print(classification_report(y_train, clf.predict(X_train)))

    print('====================================')
    print('| Area under ROC curve: %.4f |' % roc_auc_score(y_test, y_predict))
    print('| Brier loss score: %.4f |' % brier_score_loss(y_test, y_predict))
    print('====================================')
    print('\033[0m')



def train_and_validate(df, predictors, predictand, **kwargs):
    '''

    df: pd.DataFrame
    predictors: list of column names:
    predictand: column name (the one to be predicted)

    kwargs:
        - wdir,
        - test_size
        - kind: "regression" or "classification"
    '''
    wdir = kwargs.get('wdir', '.')
    figdir = os.path.join(wdir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df[predictors].values,
        df[predictand].values,
        test_size=kwargs.get('test_size', 0.3),
        random_state=kwargs.get('split_random_state', None),
        )
    X_all = df[predictors].values
    y_all = df[predictand].values

    df_train = pd.DataFrame(
        data=dict(zip(predictors, X_train.transpose())))
    df_train[predictand] = y_train


    #print('\033[1m========y_train, y_test means: %.3f, %.3f==========\033[0m' % (np.mean(y_train), np.mean(y_test)))


    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train_ = scaler.transform(X_train)
    #X_test_ = scaler.transform(X_test)
    #X_all_ = scaler.transform(X_all)

    scaler = None
    X_train_ = X_train
    X_test_ = X_test
    X_all_ = X_all

    kind = kwargs.get('kind', 'regression')

    default_models = REGRESSORS if kind == "regression" else CLASSIFIERS
    models = kwargs.get('models', default_models)

    default_print_func = print_regressor_report if kind == "regression" else print_classifier_report
    print_func = kwargs.get('print_func', default_print_func)

    predictions = {}

    for i, (name, klass) in enumerate(models.items()):
        if name in ['RandomForest', 'GradientBoosting', 'AdaBoost']:
            kwargs_obj = kwargs.get(
                "{}_options".format(name),
                kwargs.get("ensemble_options", {'n_estimators': 100}))

        else:
            kwargs_obj = kwargs.get(
                "{}_options".format(name),
                {})

        model = klass(**kwargs_obj)

        model.fit(X_train_, y_train)
        y_pred = model.predict(X_test_)

        if kind == 'regression':
            aopt, _ = curve_fit(linreg, y_pred, y_test)
            figname = os.path.join(figdir, '%s_scatter.png' % name)
            plt.figure()
            plt.plot(y_pred, y_test, 'o', alpha=0.5)
            vmin = y_test.min()
            vmax = y_test.max()
            plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
            plt.plot([vmin, vmax], linreg(np.array([vmin, vmax]), *aopt), 'b--', lw=1.5, alpha=0.8)
            plt.ylabel('Measured')
            plt.xlabel('Predicted')
            plt.title('Regression: measured = %.3f * predicted + %.3f' % (aopt[0], aopt[1]))
            plt.grid()
            plt.savefig(figname)
            plt.close()

            #figname = os.path.join(figdir, '%s_error.png' % name)
            #plt.figure()
            #plt.plot(y_test, y_pred - y_test, 'o', alpha=0.5)
            #plt.plot([vmin, vmax], [0, 0], 'k--', lw=2)
            #plt.xlabel('Measured')
            #plt.ylabel('Residuals')
            #plt.grid()
            #plt.savefig(figname)
            #plt.close()

        with open(os.path.join(wdir, '{}.pyt'.format(name)), 'wb') as pyt:
            model.fit(X_all_, y_all)
            joblib.dump({'scaler': scaler, 'model': model}, pyt)

        print_func(
            name,
            X_train, y_train,
            X_test, y_test, y_pred,
            color_code=kwargs.get('{}_color_code'.format(name), "9%i" % (i % 5 + 1)),
            predictand_name=predictand)

        #predictions[name] = model.predict(X_all_)
        predictions[name] = {'model': model,
                             'prediction': model.predict(X_all_),
                             'residuals': model.predict(X_train_) - y_train}


    return predictions, df_train


def get_optimal_mlp_size(predictors, predictand, sizes_to_test, scaler=None, **kwargs):
    import matplotlib.pyplot as plt

    wdir = kwargs.get('wdir', '.')

    X_train, X_test, y_train, y_test = train_test_split(
        predictors.values,
        predictand.values,
        test_size=kwargs.get('test_size', 0.3),
        )

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_train)

    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)

    roc_auc = []
    brier = []

    n_cases = kwargs.get('n_cases', 300)
    n_tests = len(sizes_to_test)

    for i in range(n_cases):
        roc_auc_case = []
        brier_case = []

        for sizes in sizes_to_test:
            reg = MLPRegressor(hidden_layer_sizes=sizes)
            reg.fit(X_train_, y_train)
            y_pred = reg.predict(X_test_)

            y_predict_ = np.maximum(0, np.minimum(1, y_pred))

            roc_auc_case.append(roc_auc_score(y_test, y_predict_))
            brier_case.append(brier_score_loss(y_test, y_predict_))

        roc_auc.append(roc_auc_case)
        brier.append(brier_case)

        print('Completed {:6.2f}%'.format(100 * (i + 1) / n_cases), end='\r')

    roc_auc_p90 = np.nanpercentile(roc_auc, 90, axis=0)
    roc_auc_p75 = np.nanpercentile(roc_auc, 75, axis=0)
    roc_auc_p25 = np.nanpercentile(roc_auc, 25, axis=0)
    roc_auc_p10 = np.nanpercentile(roc_auc, 10, axis=0)
    brier_p90 = np.nanpercentile(brier, 90, axis=0)
    brier_p75 = np.nanpercentile(brier, 75, axis=0)
    brier_p25 = np.nanpercentile(brier, 25, axis=0)
    brier_p10 = np.nanpercentile(brier, 10, axis=0)

    roc_auc = np.nanmean(roc_auc, axis=0)
    brier = np.nanmean(brier, axis=0)


    fname = os.path.join(wdir, 'roc_auc.png')
    plt.figure()
    plt.plot(np.arange(n_tests), roc_auc, 'g')
    plt.plot(np.arange(n_tests), roc_auc_p10, 'g-.')
    plt.plot(np.arange(n_tests), roc_auc_p25, 'g--')
    plt.plot(np.arange(n_tests), roc_auc_p75, 'g--')
    plt.plot(np.arange(n_tests), roc_auc_p90, 'g-.')
    plt.xticks(np.arange(n_tests), sizes_to_test, rotation=45)
    plt.ylabel('Area under ROC curve')
    plt.grid()
    plt.savefig(fname)
    plt.tight_layout()
    plt.close()

    fname = os.path.join(wdir, 'brier.png')
    plt.figure()
    plt.plot(np.arange(n_tests), brier, 'r')
    plt.plot(np.arange(n_tests), brier_p10, 'r-.')
    plt.plot(np.arange(n_tests), brier_p25, 'r--')
    plt.plot(np.arange(n_tests), brier_p75, 'r--')
    plt.plot(np.arange(n_tests), brier_p90, 'r-.')
    plt.xticks(np.arange(n_tests), sizes_to_test, rotation=45)
    plt.ylabel('Brier loss')
    plt.grid()
    plt.savefig(fname)
    plt.tight_layout()
    plt.close()

    iauc = np.argmax(roc_auc)
    ibrier = np.argmin(brier)

    return iauc, ibrier
