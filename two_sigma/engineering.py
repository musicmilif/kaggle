import re
import time
import math
import string
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from utils import address_map_func
string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
remove_punct_map = dict.fromkeys(map(ord, string.punctuation))


def cart2rho(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho

def cart2phi(x, y):
    phi = np.arctan2(y, x)
    return phi

def rotation_x(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return x*math.cos(alpha) + y*math.sin(alpha)

def rotation_y(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return y*math.cos(alpha) - x*math.sin(alpha)

def add_rotation(degrees, df):
    namex = "rot" + str(degrees) + "_X"
    namey = "rot" + str(degrees) + "_Y"
    df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
    df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)
    return df

def operate_on_coordinates(df):
    #polar coordinates system
    df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.7518, x["longitude"]+73.9779), axis=1)
    df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.7518, x["longitude"]+73.9779), axis=1)
    #rotations
    for angle in [15, 30, 45, 60]:
        df = add_rotation(angle, df)
    return df


class FeatureEngineer(object):
    def __init__(self):
        pass
    
    def basic(self, df):
        # Simple transform
        df["price_t"] = df["price"]/df["bedrooms"]
        df["room_sum"] = df["bedrooms"]+df["bathrooms"] 
        df["log_price"] = np.log(df['price'])
        df["log_price_t"] = df["log_price"]/df["bedrooms"]
        # Counts
        df["num_photos"] = df["photos"].apply(len)
        df["num_features"] = df["features"].apply(len)
        df["num_desc_words"] = df["description"].apply(lambda x: len(x.split(' ')))
        df["avg_desc_words"] = df["description"].apply(lambda x: len(x))/df['num_desc_words']
        # Time
        df["created"] = pd.to_datetime(df["created"])
        df["passed"] = df["created"].max() - df["created"]
        df["passed"] = df["passed"].dt.days
        df["created_year"] = df["created"].dt.year
        df["created_month"] = df["created"].dt.month
        df["created_day"] = df["created"].dt.day
        df["created_hour"] = df["created"].dt.hour
        df['Wday'] = df['created'].dt.dayofweek
        df['Yday'] = df['created'].dt.dayofyear
        df['unix_time'] = df['created'].apply(lambda x: time.mktime(x.timetuple()))
        # Price per rooms
        df['price_rooms'] = df['price']/(df['bathrooms']*0.5+df['bedrooms']+1)
        df['price_bath'] = df['price']/(df['bathrooms']+1)
        df['price_bed'] = df['price']/(df['bedrooms']+1)
        # SHOUTING!!
        df['shout'] = df['description'].apply(lambda x: sum(1 for c in x if c.isupper())/float(len(x)+1))
        return df
    
    def manager_id(self, df):
        # Label Encoding
        lbl = LabelEncoder()
        lbl.fit(list(df['manager_id'].values))
        df['encode_manager'] = lbl.transform(list(df['manager_id'].values))
        # Counts Alphabet and Numbers in manager id
        df['manager_alph'] = df['manager_id'].apply(lambda x: len(re.sub('[0-9]+', '', x)))
        df['manager_num'] = df['manager_id'].apply(lambda x: len(re.sub('[a-zA-Z]+', '', x)))
        # Number of different building_id
        tmp = df.groupby(['manager_id'])['building_id']\
                .agg(lambda x: len(x.unique()))\
                .to_frame(name='manager_build')
        df = pd.merge(df, tmp, how='left', left_on='manager_id', right_index=True)
        # Average number of bathrooms
        tmp = df.groupby(['manager_id'])['bathrooms']\
                .mean()\
                .to_frame(name='avg_bathrooms')
        df = pd.merge(df, tmp, how='left', left_on='manager_id', right_index=True)
        # Average number of bedrooms
        tmp = df.groupby(['manager_id'])['bedrooms']\
                .mean()\
                .to_frame(name='avg_bedrooms')
        df = pd.merge(df, tmp, how='left', left_on='manager_id', right_index=True)
        # Average of prices
        tmp = df.groupby(['manager_id'])['price']\
                .mean()\
                .to_frame(name='avg_price')
        df = pd.merge(df, tmp, how='left', left_on='manager_id', right_index=True)
        tmp = df.groupby(['manager_id'])['log_price']\
                .mean()\
                .to_frame(name='avg_log_price')
        df = pd.merge(df, tmp, how='left', left_on='manager_id', right_index=True)
        # Average of num_description_words
        tmp = df.groupby(['manager_id'])['num_desc_words']\
                .mean()\
                .to_frame(name='avg_num_desc')
        df = pd.merge(df, tmp, how='left', left_on='manager_id', right_index=True)
        return df

    def location(self, df):
        df['address1'] = df['display_address'].apply(lambda x: x.lower())\
                                            .apply(lambda x: x.translate(remove_punct_map))\
                                            .apply(lambda x: address_map_func(x))

        new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']
        for col in new_cols:
            df[col] = df['address1'].apply(lambda x: 1 if col in x else 0)
        df['other_address'] = df[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
        # Using k-means to cluster locations
        kmeanscluster = KMeans(n_clusters=40)
        tmp = df[['longitude', 'latitude']].copy()
        tmp['longitude'] = (tmp['longitude']-tmp['longitude'].mean())/tmp['longitude'].std()
        tmp['latitude'] = (tmp['latitude']-tmp['latitude'].mean())/tmp['latitude'].std()
        kmeanscluster.fit(tmp[['longitude', 'latitude']])
        df['kmeans_neighbor'] = kmeanscluster.labels_
        # Nearby count for each location
        df["pos"] = df['longitude'].round(3).astype(str) + '_' + df['latitude'].round(3).astype(str)
        vals = df['pos'].value_counts()
        dvals = vals.to_dict()
        df["density"] = df['pos'].apply(lambda x: dvals.get(x, vals.min()))
        # each building_id count only once
        dvals = df[['building_id', 'pos']].drop_duplicates()['pos']\
                                          .value_counts()\
                                          .to_dict()
        df["density_u"] = df['pos'].apply(lambda x: dvals.get(x, vals.min()))
        # Duplicate address
        tmp = df.groupby(['building_id'])['pos']\
                .apply(lambda x: len(set(x)))\
                .to_frame(name='dup_address')
        df = pd.merge(df, tmp, how='left', left_on='building_id', right_index=True)
        # Coordinates
        df = operate_on_coordinates(df)

        return df


def target_encoder(train_df, test_df):
    index = list(range(train_df.shape[0]))
    random.shuffle(index)

    a, b, c = [np.nan]*len(train_df), [np.nan]*len(train_df), [np.nan]*len(train_df)
    for i in range(5):
        building_level = {}
        for j in train_df['manager_id'].values:
            building_level[j]=[0, 0, 0]
        
        #select the fifth part as the validation set, and the other as the train set
        test_index = index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index = list(set(index).difference(test_index))
        #sum up the count of each level for a specific manager
        for j in train_index:
            tmp = train_df.iloc[j]
            if tmp['interest_level'] == 'low':
                building_level[tmp['manager_id']][0] += 1
            if tmp['interest_level'] == 'medium':
                building_level[tmp['manager_id']][1] += 1
            if tmp['interest_level'] == 'high':
                building_level[tmp['manager_id']][2] += 1
        for j in test_index:
            tmp=train_df.iloc[j]
            if sum(building_level[tmp['manager_id']])!=0:
                a[j] = building_level[tmp['manager_id']][0]*1.0/sum(building_level[tmp['manager_id']])
                b[j] = building_level[tmp['manager_id']][1]*1.0/sum(building_level[tmp['manager_id']])
                c[j] = building_level[tmp['manager_id']][2]*1.0/sum(building_level[tmp['manager_id']])
    train_df['manager_level_low'] = a
    train_df['manager_level_medium'] = b
    train_df['manager_level_high'] = c
    train_df['manager_skill'] = (np.asarray(c)*2+np.asarray(b))/train_df['manager_build']

    a, b, c = [], [], []
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    for j in range(train_df.shape[0]):
        tmp = train_df.iloc[j]
        if tmp['interest_level'] == 'low':
            building_level[tmp['manager_id']][0] += 1
        if tmp['interest_level'] == 'medium':
            building_level[tmp['manager_id']][1] += 1
        if tmp['interest_level'] == 'high':
            building_level[tmp['manager_id']][2] += 1
    for i in test_df['manager_id'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df['manager_level_low'] = a
    test_df['manager_level_medium'] = b
    test_df['manager_level_high'] = c
    test_df['manager_skill'] = (np.asarray(c)*2+np.asarray(b))/test_df['manager_build']
    train_df['features'] = train_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    test_df['features'] = test_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))

    return train_df, test_df