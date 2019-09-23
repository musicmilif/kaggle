import gc
import re
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from nltk.corpus import stopwords

import lightgbm as lgb


sys.path.insert(0, '../input/wordbatch/wordbatch/')
NUM_BRANDS = 4500
NUM_CATEGORIES = 1250
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)
    dataset['item_description'].loc[dataset['item_description']=='No description yet'] = 'missing'


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u' '.join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


def main():
    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')

    print('Finished to load data')
    nrow_test = train.shape[0]
    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    submission: pd.DataFrame = test[['test_id']]

    del train, test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('Split categories completed.')

    handle_missing_inplace(merge)
    print('Handle missing completed.')

    cutting(merge)
    print('Cut completed.')

    to_categorical(merge)
    print('Convert categorical completed')

    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name_cv = cv.fit_transform(merge['name'])
    
    cv = CountVectorizer()
    X_category1_cv = cv.fit_transform(merge['general_cat'])
    X_category2_cv = cv.fit_transform(merge['subcat_1'])
    X_category3_cv = cv.fit_transform(merge['subcat_2'])
    
    wb = wordbatch.WordBatch(normalize_text, 
                             extractor=(WordBag, {'hash_ngrams': 2, 
                                                  'hash_ngrams_weights': [1.5, 1.0],
                                                  'hash_size': 2 ** 29, 
                                                  'norm': None, 
                                                  'tf': 'binary', 
                                                  'idf': None,
                                                  }), 
                             procs=8)
    wb.dictionary_freeze= True
    X_name = wb.fit_transform(merge['name'])
    del wb
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('Vectorize `name` completed.')

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])
    print('Count vectorize `categories` completed.')

    wb = wordbatch.WordBatch(normalize_text, 
                             extractor=(WordBag, {'hash_ngrams': 2, 
                                                  'hash_ngrams_weights': [1.0, 1.0],
                                                  'hash_size': 2 ** 28, 
                                                  'norm': 'l2', 
                                                  'tf': 1.0,
                                                  'idf': None})
                             , procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(merge['item_description'])
    del wb
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), 
                                              dtype=bool)]
    print('Vectorize `item_description` completed.')

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('Label binarize `brand_name` completed.')

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('Get dummies on `item_condition_id` and `shipping` completed.')
    
    num_chars = merge['item_description'].apply(lambda x: len(x)).values
    num_words = merge['item_description'].apply(lambda x: len(x.split(' '))).values
    num_upper = merge['item_description'].apply(lambda x: len(re.findall('[A-Z]+', x))).values
    num_chars = num_chars / max(num_chars)
    num_words = num_words / max(num_words)
    num_upper = num_upper / max(num_upper)
    
    X_feature = np.vstack([num_chars, num_words, num_upper]).T
    print('musicmilif features completed.')
    
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name, 
                           X_category1_cv, X_category2_cv, X_category3_cv, X_name_cv, X_feature)).tocsr()
    print('Create sparse merge completed')
    del X_dummies, X_description, X_brand, X_category1, X_category2, X_category3 
    del X_name, X_category1_cv, X_category2_cv, X_category3_cv, X_name_cv, X_feature 
    del num_chars, num_words, num_upper
    gc.collect()

    # Remove features with document frequency <=1
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]

    gc.collect()
    
    train_X, train_y = X, y

    model = Ridge(solver='auto', fit_intercept=True, alpha=5.0,max_iter=100, normalize=False, tol=0.05)
    model.fit(train_X, train_y)
    print('Train Ridge completed')
    predsR = model.predict(X_test)
    print('Predict Ridge completed')

    model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)
    model.fit(train_X, train_y)
    print('Train FTRL completed')
    predsF = model.predict(X_test)
    print('Predict FTRL completed')

    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)
    model.fit(train_X, train_y)
    print('Train FM_FTRL completed')
    predsFM = model.predict(X_test)
    print('Predict FM_FTRL completed')

    params = {
        'learning_rate': 0.6,
        'application': 'regression',
        'max_depth': 9,
        'num_leaves': 24,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.9,
        'bagging_freq': 6,
        'feature_fraction': 0.8,
        'nthread': 4,
        'min_data_in_leaf': 51,
        'max_bin': 64
    }

    # Remove features with document frequency <=200
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 200, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]

    train_X, train_y = X, y
    d_train = lgb.Dataset(train_X, label=train_y)
    watchlist = [d_train]
    model = lgb.train(params, train_set=d_train, num_boost_round=1800, valid_sets=watchlist,
                      early_stopping_rounds=500, verbose_eval=400)

    predsL = model.predict(X_test)
    print('Predict LGBM completed')

    preds = (predsR * 1+ predsF * 1 + predsFM * 16 + predsL * 6) / (1+1+16+6)
    submission['price'] = np.expm1(preds)
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()