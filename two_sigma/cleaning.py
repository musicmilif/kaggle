import pandas as pd


class Cleaner(object):
    def __init__(self):
        pass
    
    def train(self, df):
        df["price"] = df["price"].clip(upper=13000)
        return df

    def test(self, df):
        df["bathrooms"].loc[19671] = 1.5
        df["bathrooms"].loc[22977] = 2.0
        df["bathrooms"].loc[63719] = 2.0
        return df

    def magic(self, df):
        df.loc[80240,"time_stamp"] = 1478129766 
        df["magic_date"] = pd.to_datetime(df["magic"], unit="s")
        df["magic_passed"] = (df["magic_date"].max() - df["magic_date"]).astype("timedelta64[D]").astype(int)
        df["magic_month"] = df["magic_date"].dt.month
        df["magic_week"] = df["magic_date"].dt.week
        df["magic_day"] = df["magic_date"].dt.day
        df["magic_dayofweek"] = df["magic_date"].dt.dayofweek
        df["magic_dayofyear"] = df["magic_date"].dt.dayofyear
        df["magic_hour"] = df["magic_date"].dt.hour
        df["magic_monthBeginMidEnd"] = df["magic_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)
        return df
    
    def merge(train_df, test_df, magic_df):
        df = pd.concat([train_df, test_df], axis=0)
        df = pd.merge(df, magic_df, on='listing_id')
        return df