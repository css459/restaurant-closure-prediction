#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# transform.py
#
# Utility functions for train / test splitting, re-balancing,
# and other numerical preparations for data matrices
#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#
# Split
#

def split_train_test(df, y, split_size=0.2):
    """
    Performs preprocessing on the master dataset
    and splits it into training and test sets, treating
    it as time series data.

    :param df:          The master dataset
    :param y:           Column to predict on
    :param split_size:  Percent of data to use for testing
    :return:            X/Y, Train/Test Sets
    """
    if split_size <= 0:
        return df.drop(y, 1), df[y], None, None

    y_set = df[y]
    x_set = df.drop(y, 1)

    return train_test_split(x_set, y_set, train_size=1 - split_size, random_state=42)

#
# Resampling
#

# def selective_master_downsample(master):
#     """
#     Selectively downsample sparse data
#     in master dataset
#
#     :param master:  The master dataset
#     :return:        `DataFrame`
#     """
#
#     return master.dropna()


#
# Value Scaling
#

def normalize_values(df, y):
    """
    Normalize columns of values in DataFrame

    :param df:  DataFrame
    :param y:   Name of Y col as string
    :return:    DataFrame
    """
    cols = df.drop(y, 1).columns
    return pd.DataFrame(StandardScaler().fit_transform(df.drop(y, 1)), columns=cols), df[y]


def min_max_scale_values(df, y):
    """
    Min/Max scale columns of values in DataFrame

    :param df:  DataFrame
    :param y:   Name of Y col as string
    :return:    DataFrame
    """
    if y:
        cols = df.drop(y, 1).columns
        return pd.DataFrame(MinMaxScaler().fit_transform(df.drop(y, 1)), columns=cols), df[y]
    else:
        cols = df.columns
        return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=cols)
