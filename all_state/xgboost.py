import re
import os
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import skew, boxcox
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i])-ord('A')+1)*26**(ln-i-1)
    return r


def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds-labels)
    den = abs(x) + c
    grad, hess = c*x/(den), c*c/(den*den)
    return grad, hess


def mae4xgb(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift, np.exp(yhat)-shift)


def makeunskew(train, test, num_col):
    train_nrows = len(train)
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[num_col].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, train_nrows


def main():
    shift = 200
    c = 2
    n_folds = 10
    cv_sum = 0

    COMB_FEATURE = '''
                    cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,
                    cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,
                    cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,
                    cat82,cat25
                   '''
    COMB_FEATURE = re.sub('\s+', '', COMB_FEATURE).split(',')

    data_dir = '/disk/project_data/cht/'
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    ids = test_df['id']

    num_col = [col for col in train_df.columns if 'cont' in col]
    cat_col = [col for col in train_df.columns if 'cat' in col]
    train_test_df, train_nrows = makeunskew(train_df, test_df, num_col)
    print('Merge train and test completed.')

    for comb in itertools.combinations(COMB_FEATURE, 2):
        col = '_'.join(comb)
        train_test_df[col] = train_test_df[comb[0]] + train_test_df[comb[1]]
        train_test_df[col] = train_test_df[col].apply(encode)
    print('Combine high correlated categorical features completed.')
        
    cat_col = [col for col in train_df.columns if 'cat' in col]
    for col in cat_col:
        train_test_df[col] = train_test_df[col].apply(encode)
    print('Label encoding completed.')

    ss = StandardScaler()
    train_test_df[num_col] = ss.fit_transform(train_test_df[num_col].values)
    train_df = train_test_df.iloc[:train_nrows].copy()
    test_df = train_test_df.iloc[train_nrows:].copy()

    train_y = np.log(train_df['loss'] + shift)
    train_X = train_df.drop(['loss', 'id'], axis=1)
    test_X = test_df.drop(['loss', 'id'], axis=1)


    xgb_rounds = []

    d_train_full = xgb.DMatrix(train_X, label=train_y)
    d_test = xgb.DMatrix(test_X)

    kf = KFold(n_splits=n_folds)
    for i, (train_idx, valid_idx) in enumerate(kf.split(train_X)):
        print('\n Fold %d' % (i+1))
        X_train, X_valid = train_X.iloc[train_idx], train_X.iloc[valid_idx]
        y_train, y_valid = train_y.iloc[train_idx], train_y.iloc[valid_idx]

        params = {
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.03,
            'objective': 'reg:linear',
            'max_depth': 12,
            'min_child_weight': 100,
            'booster': 'gbtree',
        }

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        clf = xgb.train(params, d_train, 100000, watchlist, early_stopping_rounds=100, obj=fair_obj, 
                        feval=mae4xgb, verbose_eval=100)

        xgb_rounds.append(clf.best_iteration)
        y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit)) - shift

        pred = pred + y_pred if i > 0 else y_pred
        cv_sum += clf.best_score

    mpred = pred / n_folds
    score = cv_sum / n_folds
    print('Average eval-MAE: {:.6f}'.format(score))
    n_rounds = int(np.mean(xgb_rounds))

    result = pd.DataFrame(mpred, columns=['loss'])
    result['id'] = ids
    result = result.set_index('id')
    result.to_csv(os.path.join(data_dir, 'xgboost_sub.csv'), index=True, index_label='id')


if __name__ == '__main__':
    main()
