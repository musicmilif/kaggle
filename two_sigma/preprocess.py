import os
import logging
import numpy as  np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cleaning import Cleaner
from engineering import FeatureEngineer, target_encoder


DATA_DIR = './data'
logger = logging.getLogger(__name__)


def load_data(data_path):
    train_df = pd.read_json(os.path.join(data_path, 'train.json'))
    test_df = pd.read_json(os.path.join(data_path, 'test.json'))
    magic_df = pd.read_csv(os.path.join(data_path, 'listing_image_time.csv'), 
                           usecols=['listing_id', 'magic'])
    
    return (train_df, test_df, magic_df)


def pre_process():
    cleaning = Cleaner()
    engineering = FeatureEngineer()

    # Cleaning
    train_df, test_df, magic_df = load_data(DATA_DIR)
    train_df = train_df.iloc[:300]
    test_df = test_df.iloc[:100]
    len_train = len(train_df)
    train_df = cleaning.train(train_df)
    test_df = cleaning.test(test_df)
    magic_df = cleaning.magic(magic_df)
    logger.info('Finished cleaning...')

    # Feature Engineering
    df = pd.concat([train_df, test_df], sort=False)
    df = pd.merge(df, magic_df, on='listing_id')
    df = engineering.basic(df)
    df = engineering.manager_id(df)
    df = engineering.location(df)
    logger.info('Finished feature engineering...')

    # Target Encoding
    train_df, test_df = df.iloc[:len_train], df.iloc[len_train:]
    train_df, test_df = target_encoder(train_df, test_df)
    logger.info('Finished target encoding...')
    
    return train_df, test_df
