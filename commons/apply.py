import os
import joblib

import pandas as pd

from traceback import print_exc


def write_pred(ids, y_pred, fname, id_name="Id", prediction_name="Prediction"):
    df = pd.DataFrame(data={prediction_name: y_pred.astype(int), id_name: ids})
    df = df.set_index(id_name)
    df.index.name = id_name
    df.to_csv(fname)


def apply_models(ids, predictors, **kwargs):
    '''

    predictors: pd.DataFrame

    kwargs:
        - wdir,
        - test_size
        - kind: "regression" or "classification"
    '''
    models_dir = kwargs.get('models_dir', '.')
    output_dir = kwargs.get('output_dir', '.')
    os.makedirs(output_dir, exist_ok=True)

    prediction_name = kwargs.get('predictand', 'Survived')
    id_name = kwargs.get('id_name', 'Id')

    X_all = predictors.values

    predictions = {}

    for f in os.listdir(models_dir):
        if f.endswith('.pyt'):
            dat = joblib.load(os.path.join(models_dir, f))

        #scaler = dat['scaler']
        model = dat['model']
        model_name = f.replace('.pyt', '')

        name = f.replace('.pyt', '')

        print('Applying model: \033[1m%s\033[0m' % name)

        #X_all_ = scaler.transform(X_all)
        X_all_ = X_all.copy()
        try:
            y_pred = model.predict(X_all_)
        except:
            print('\t\033[91mFailed.\033[0m')
            print_exc()
            continue

        write_pred(
            ids, y_pred, os.path.join(output_dir, "%s.csv" % name),
            id_name=id_name,
            prediction_name=prediction_name)

        df_pred = predictors.copy()
        df_pred[prediction_name] = y_pred
        df_pred[id_name] = ids
        predictions[model_name] = df_pred

    return predictions
