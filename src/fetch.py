#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# fetch.py
#

# This file handles the fetching of raw CSV data from
# the `data/` directory. These functions will then be
# used in `merge.py` and `preprocess.py` to combine and
# impute their values

import os
import pandas as pd


#
# Constants
#


DATA_DIR = "data/"
NYC_OPEN_DATA_DIR = DATA_DIR + "nyc_open_data/datasets/"


#
# Helpers
#


def _strip_strings(df):
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.lower())
    return df


def _camel_case_cols(df):
    cols = df.columns
    cols = [c.lower().replace(" ", "_") for c in cols]
    df.columns = cols
    return df


def _remove_punctuation(df):
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.replace(r'[^\w\s]', ''))
    return df


#
# NYC Open Data Fetching
#


def fetch_inspection_data(append_closure_col=False):
    df = pd.read_csv(NYC_OPEN_DATA_DIR + "Inspections.csv")

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    # Filter by restaurants only
    industries = ['sidewalk cafe  013', 'gaming cafe  129', 'restaurant  818']
    df = df.loc[df['industry'].isin(industries)]

    # Drop unneeded columns
    drop_cols = ['industry', 'unit_type', 'unit', 'description']
    df = df.drop(drop_cols, 1)

    # Append closure column
    if append_closure_col:
        pass

    return df


def fetch_restaurant_inspection_data():
    df = pd.read_csv(NYC_OPEN_DATA_DIR + 'DOHMH_New_York_City_Restaurant_Inspection_Results.csv')

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    # # Filter by restaurants only
    # industries = ['sidewalk cafe  013', 'gaming cafe  129', 'restaurant  818']
    # df = df.loc[df['industry'].isin(industries)]
    #
    # # Drop unneeded columns
    # drop_cols = ['industry', 'unit_type', 'unit', 'description']
    # df = df.drop(drop_cols, 1)

    return df


def fetch_restaurant_violation_lookup_table(refresh=False):

    if refresh or not os.path.isfile(NYC_OPEN_DATA_DIR + 'violations.csv'):
        df = fetch_restaurant_inspection_data()
        v = df.groupby(['violation_code', 'violation_description'], as_index=False).size().reset_index()
        v.columns = ['violation_code', 'violation_description', 'count']
        v.to_csv(NYC_OPEN_DATA_DIR + 'violations.csv', index=False)

    return pd.read_csv(NYC_OPEN_DATA_DIR + 'violations.csv')


def fetch_legally_operating_businesses():
    pass
