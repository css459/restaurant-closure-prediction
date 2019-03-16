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
import numpy as np
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


def fetch_inspection_data(append_closure_col=True):
    """
    This Dataset can be used to find HARD CLOSURES, those which
    are not able to even conduct an inspection.
    :param append_closure_col:
    :return:
    """
    df = pd.read_csv(NYC_OPEN_DATA_DIR + "Inspections.csv")

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    # Filter by restaurants only
    industries = ['sidewalk cafe  013', 'gaming cafe  129', 'restaurant  818']
    df = df.loc[df['industry'].isin(industries)]

    # Fix date
    df['inspection_year'] = df['inspection_date'].str[-4:]
    df['inspection_month'] = df['inspection_date'].str[:2]
    df['inspection_day'] = df['inspection_date'].str[2:4]

    # Drop unneeded columns
    drop_cols = ['industry', 'unit_type', 'unit', 'description', 'inspection_date']
    df = df.drop(drop_cols, 1)

    # Append closure column
    if append_closure_col:
        # A *closure* is determined by the following inspection results
        closures = ['out of business', 'unable to locate', 'fail']
        df['is_closed'] = np.where(df['inspection_result'].isin(closures), 1, 0)

        # Sort since we will then drop duplicates
        df = df.sort_values(by='is_closed', ascending=False)

    df = df.drop_duplicates('record_id', keep='first')
    df = df.sort_values(by=['inspection_year', 'inspection_month', 'inspection_day'], ascending=True)

    return df


def fetch_restaurant_inspection_data():
    """
    This Dataset can be used for finding SOFT CLOSURES: Those which are caused
    by inspections.
    :return:
    """
    df = pd.read_csv(NYC_OPEN_DATA_DIR + 'DOHMH_New_York_City_Restaurant_Inspection_Results.csv')

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    # Clean grade column

    # Grades in ascending order of quality
    grades = ['c', 'b', 'a']

    # Map grades to index values plus 1 (i.e: an "A" is a 3)
    df['grade'] = df['grade'].apply(lambda x: grades.index(x) + 1 if x in grades else np.nan)
    df['grade'] = pd.to_numeric(df['grade'])

    # Categorize violation action by open or closed
    # and count violations

    # Dictionary of String : (is_closed, violation_count)
    action_lookup = {
        'violations were cited in the following areas': (False, 1),
        'no violations were recorded at the time of this inspection': (False, 0),
        'establishment closed by dohmh  violations were cited in the following areas and those requiring immediate action were addressed': (
            True, 1),
        'establishment reclosed by dohmh': (True, 1),
        'establishment reopened by dohmh': (False, 0),
        np.nan: (False, 0)
    }

    # Apply action mapping
    df['is_closed'] = df['action'].apply(lambda x: action_lookup[x][0])

    # (Violation count is defined as the number of inspections which *resulted*
    # in a violation or closure)
    df['violation_count'] = df['action'].apply(lambda x: action_lookup[x][1])

    # Fix date
    df['inspection_year'] = df['inspection_date'].str[-4:]
    df['inspection_month'] = df['inspection_date'].str[:2]
    df['inspection_day'] = df['inspection_date'].str[2:4]

    # Fix critical flag
    df['critical_flag'] = np.where(df['critical_flag'] == 'critical', 1, 0)
    df['critical_flag'] = pd.to_numeric(df['critical_flag'])

    # Drop unneeded cols
    drop_cols = ['action', 'violation_code', 'violation_description', 'inspection_date',
                 'grade_date', 'record_date', 'inspection_type']
    df = df.drop(drop_cols, 1)

    # Sort by date
    df = df.sort_values(by=['inspection_year', 'inspection_month', 'inspection_day'], ascending=False)

    # Group by identifiers
    df['total_inspections'] = 1
    ids = ['camis', 'dba', 'boro', 'building', 'street', 'zipcode', 'phone', 'cuisine_description']
    g = df.groupby(ids, as_index=False)

    # The aggregation will occur as follows:
    #   is_closed: latest value by date
    #   violation_count: sum
    #   critical_flag: sum
    #   score: average
    #   grade: average
    agr = {'critical_flag': np.sum,
           'score': np.mean,
           'grade': np.mean,
           'violation_count': np.sum,
           'total_inspections': np.sum}
    ga = g.agg(agr)

    # The date and is_closed will be the latest value
    g = g.first()

    # Apply aggregations to most recent entries for restaurants
    g[list(agr.keys())] = ga[list(agr.keys())]

    # Make violation ratio
    g['violation_ratio'] = g['violation_count'] / g['total_inspections']

    return g


def fetch_restaurant_violation_lookup_table(refresh=False):
    if refresh or not os.path.isfile(NYC_OPEN_DATA_DIR + 'violations.csv'):
        df = fetch_restaurant_inspection_data()
        v = df.groupby(['violation_code', 'violation_description'], as_index=False).size().reset_index()
        v.columns = ['violation_code', 'violation_description', 'count']
        v.to_csv(NYC_OPEN_DATA_DIR + 'violations.csv', index=False)

    return pd.read_csv(NYC_OPEN_DATA_DIR + 'violations.csv')


def fetch_legally_operating_businesses():
    df = pd.read_csv(NYC_OPEN_DATA_DIR + 'Legally_Operating_Businesses.csv')

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    return df
