import warnings
warnings.filterwarnings('ignore')

import re
import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    DATA_DIR = '/disk/landmark_rec/'
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    # Remove download failed datas
    all_ids = [re.sub('.jpg', '', os.path.basename(file_id)) for file_id in glob.glob(os.path.join(DATA_DIR, 'train/*.jpg'))]
    train_df = train_df.loc[train_df['id'].isin(all_ids)]
    print('Original training data size: {0}'.format(len(train_df)))

    # Dealing with imbalanced data and set rare class to unknown
    label_cnt = train_df['landmark_id'].value_counts()
    train_df['landmark_id'].loc[train_df['landmark_id'].isin(label_cnt[label_cnt<=1].index)] = 99999
    print('Set rare events to `landmark_id`=99999')

    label_cnt = train_df['landmark_id'].value_counts()
    for lm_id in label_cnt[label_cnt > 200].index:
        idx = train_df.loc[train_df['landmark_id']==lm_id].index
        rand_idx = np.random.choice(len(idx), size=min([len(idx)-199, int(len(idx)*0.99)]), replace=False)
        idx = idx[rand_idx]
        train_df.drop(idx, inplace=True)
    print('Drop samples completed.')

    for lm_id in label_cnt[label_cnt < 30].index:
        idx = train_df.loc[train_df['landmark_id']==lm_id].index
        rand_idx = np.random.choice(len(idx), size=min([int(len(idx)*10), 30]), replace=True)
        idx = idx[rand_idx]
        train_df = pd.concat([train_df, train_df.loc[idx]])
    print('Bootstrap completed.')

    train_df.reset_index(drop=True, inplace=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_df.to_csv(os.path.join(DATA_DIR, 'train_balanced.csv'), index=False)
    print('After resample training data size: {0}'.format(len(train_df)))


if __name__ == '__main__':
    main()
