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
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#
# Split
#

def split_master_train_test(master, y="total_closures"):
    """
    Performs preprocessing on the master dataset
    and splits it into training and test sets, treating
    it as time series data.

    :param master:  The master dataset
    :param y:       Column to predict on
    :return:        X/Y, Train/Test Sets
    """
    pass


def split_restaurant_train_test(restaurant_inspections, y="is_closed"):
    """
    Performs preprocessing on the Restaurant Inspections
    dataset and splits it into training and testing sets,
    treating each row as a unique restaurant.

    :param restaurant_inspections:  The Restaurant Inspections dataset
    :param y:       Column to predict on
    :return:                        X/Y. Train/Test Sets
    """
    pass


#
# Resampling
#

def selective_master_downsample(master):
    """
    Selectively downsample sparse data
    in master dataset

    :param master:  The master dataset
    :return:        `DataFrame`
    """

    return master.dropna()


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
