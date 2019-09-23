import os
import gc

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def train_generator(X, y, batch_size, shuffle=True):
    while True:
        for start in range(0, X.shape[0], batch_size):
            end = min(start+batch_size, X.shape[0])
            sample_idx = np.arange(start, end)       
            if shuffle:
                np.random.shuffle(sample_idx)
            x_batch = X[sample_idx, :].toarray()
            y_batch = y[sample_idx]
            
            yield x_batch, y_batch

            
def valid_generator(X, batch_size, shuffle=False):
    while True:
        for start in range(0, X.shape[0], batch_size):
            end = min(start+batch_size, X.shape[0])
            sample_idx = np.arange(start, end)       
            if shuffle:
                np.random.shuffle(sample_idx)
            x_batch = X[sample_idx, :].toarray()

            yield x_batch


def nn_model():
    model = Sequential()

    model.add(Dense(400, input_dim=X_train.shape[1], kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(200, kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='he_normal'))
    model.compile(loss='mae', optimizer='adadelta')
    
    return (model)


def main():
    shift = 200
    n_bags = 1
    n_epochs = 55
    n_folds = 5
    batch_size = 128
    data_dir = '/disk/project_data/cht/'
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test_df['loss'] = np.nan

    train_y = np.log(train_df['loss'].values + shift)
    id_train = train_df['id'].values
    id_test = test_df['id'].values

    train_nrows = len(train_df)
    train_test_df = pd.concat((train_df, test_df), axis=0)

    ## Preprocessing and transforming to sparse data
    sparse_data = []

    num_col = [col for col in train_test_df.columns if 'cont' in col]
    cat_col = [col for col in train_test_df.columns if 'cat' in col]

    for col in cat_col:
        dummy = pd.get_dummies(train_test_df[col].astype('category'))
        tmp = csr_matrix(dummy)
        sparse_data.append(tmp)

    scaler = StandardScaler()
    tmp = csr_matrix(scaler.fit_transform(train_test_df[num_col]))
    sparse_data.append(tmp)
    sparse_data = hstack(sparse_data, format='csr')

    train_X = sparse_data[:train_nrows, :]
    test_X = sparse_data[train_nrows:, :]

    del train_test_df, train_df, test_df, tmp, sparse_data
    gc.collect()


    kf = KFold(n_splits=n_folds)

    pred_oob = np.zeros(train_X.shape[0])
    pred_test = np.zeros(test_X.shape[0])

    for i, (train_idx, valid_idx) in enumerate(kf.split(train_X)):
        X_train, X_valid = train_X[train_idx], train_X[valid_idx]
        y_train, y_valid = train_y[train_idx], train_y[valid_idx]

        preds, pred_test = np.array([]), np.array([])

        model = nn_model()
        fit = model.fit_generator(generator=train_generator(X_train, y_train, batch_size), 
                                  steps_per_epoch=np.ceil(train_X.shape[0]/batch_size), epochs=1, verbose=1)
        
        
        for start in range(0, X_valid.shape[0], batch_size):
            end = min(start + batch_size, X_valid.shape[0])
            x_batch = X_valid[start:end, :].toarray()
            tmp_preds = model.predict_on_batch(x_batch)[:, 0]
            preds = np.append(preds, tmp_preds)
        preds = np.exp(preds) - shift

        for start in range(0, test_X.shape[0], batch_size):
            end = min(start + batch_size, test_X.shape[0])
            x_batch = test_X[start:end, :].toarray()
            tmp_preds = model.predict_on_batch(x_batch)[:, 0]
            pred_test = np.append(pred_test, tmp_preds)
        pred_test = np.exp(pred_test) - shift    
                        
        preds /= n_bags
        pred_oob[valid_idx] = preds
        score = mean_absolute_error(np.exp(y_valid) - shift, preds)
        print('Fold {0} - MAE: {1}'.format(i, score))
    print('Total - MAE: {0}'.format(mean_absolute_error(np.exp(train_y) - shift, pred_oob)))

    df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
    df.to_csv(os.path.join(data_dir, 'preds_oob.csv'), index=False)
    pred_test /= (n_folds * n_bags)
    df = pd.DataFrame({'id': id_test, 'loss': pred_test})
    df.to_csv(os.path.join(data_dir, 'submission_keras.csv'), index=False)


if __name__ == '__main__':
    main()
