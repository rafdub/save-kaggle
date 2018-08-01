#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Run trained models on the test sample

'''

from mlcase_fit import load_sample, clean_sample, get_predictors_and_encoders, ROOT_DIR, DATA_DIR, MODELS_DIR, ENCODERS_FILE, SCALER_FILE, SortedLabelEncoder, univariate_sample_analysis, CATEGORICAL_FEATURES, BOOL_COLS

import pandas as pd
import numpy as np
import joblib
import shutil
import os


OUTPUT_TEMPLATE = os.path.join(DATA_DIR, 'ml_case_test_output_template.csv')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')


def write_churn_prediction(fname, df_predictions):
    df = pd.read_csv(OUTPUT_TEMPLATE).set_index('id')
    df['Churn_prediction'] = df_predictions['prediction']
    df['Churn_probability'] = df_predictions['probability']
    df = df.sort_values('Churn_probability', ascending=False)
    del df['Unnamed: 0']
    df.reset_index(level=0, inplace=True)

    # Drop id to a column
    df.to_csv(fname)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Run one of the trained machine learning algorithms')

    parser.add_argument(
        'model_name',
        type=str,
        help='''name of the regressor to use
        do not use classifiers as they cannot provide probability''')

    parser.add_argument(
        '--force',
        action='store_true',
        help='''Force the algorithm to use the .csv files
        instead of previously loaded data''')

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='''Analyze the data (on a random churn rate of 10 percent,
        just to make the plotting functions work''')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_sample('test', force=args.force)
    df = clean_sample(df)

    model_file = os.path.join(MODELS_DIR, '%s.pyt' % args.model_name)
    if not os.path.isfile(model_file):
        raise FileNotFoundError('Model file \033[91m%s\033[0m does not exist' % model_file)
    model = joblib.load(model_file)

    encoders = joblib.load(ENCODERS_FILE)
    predictors, encoders = get_predictors_and_encoders(df, encoders=encoders)

    if args.analyze:
        predictors_for_plots = predictors.copy()
        predictors_for_plots['churn'] = (np.random.random(len(predictors)) > 0.1).astype(int)

        univariate_sample_analysis(
            predictors_for_plots,
            os.path.join(RESULTS_DIR, 'analysis'),
            categorical_features=CATEGORICAL_FEATURES+BOOL_COLS,
            verbose=1)


    if 'mlp' in args.model_name:
        scaler = joblib.load(SCALER_FILE)
        test_sample = scaler.transform(predictors.values)

    else:
        test_sample = predictors.values

    probability = model.predict(test_sample)


    predictors['probability'] = np.minimum(1, np.maximum(0, probability))
    predictors['prediction'] = (probability > 0.5).astype(int)

    print('Predicted a churn rate of %.3f' % predictors['prediction'].mean())

    fname = os.path.join(RESULTS_DIR, 'results_%s.csv' % args.model_name)
    write_churn_prediction(fname, predictors)
