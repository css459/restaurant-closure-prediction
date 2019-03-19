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
IRS_DATA_DIR = DATA_DIR + "irs/datasets/"
NYC_OPEN_DATA_DIR = DATA_DIR + "nyc_open_data/datasets/"
YAHOO_FINANCE = DATA_DIR + "yahoo_finance/"


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


def fetch_alternative_agi_returns(as_percents=True):
    df = pd.read_csv(IRS_DATA_DIR + "AGI-Returns.csv")

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    # Filter out non-NYC zip codes for faster processing
    df = df.loc[df['zip'] < 11500]

    # Pivot table
    df = df.join(pd.get_dummies(df['size_of_adjusted_gross_income'], prefix='agi'))

    agi_cols = [c for c in df.columns.tolist() if "agi_" in c]

    for c in agi_cols:
        df[c] = np.where(df[c] == 1, df['number_of_returns'], 0)

    # Drop cols pre-pivot
    df = df.drop(['size_of_adjusted_gross_income', 'number_of_returns'], 1)

    # Make this table numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], downcast='float')

    # Make each row a zip
    df = df.groupby('zip', as_index=False).sum()
    df = _camel_case_cols(df)

    # Make each column into a percent representation of
    # the whole.
    # Appends "total tax returns" column
    if as_percents:
        agi_cols = [c for c in df.columns.tolist() if "agi_" in c]
        df['total_tax_returns'] = df[agi_cols].sum(axis=1)
        df[agi_cols] = df[agi_cols].div(df['total_tax_returns'], axis=0)

    return df


def fetch_alternative_demographic_stats_data():
    df = pd.read_csv(NYC_OPEN_DATA_DIR + "Demographic_Statistics_By_Zip_Code.csv")

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    df = df.rename(columns={'jurisdiction_name': 'zip'})

    # Filter out non-NYC zip codes for faster processing
    df = df.loc[df['zip'] < 11500]

    # Drop the unneeded, more specific count columns
    drop_cols = [c for c in df.columns if "count" in c and c != 'count_participants']

    # Plus sparse ones
    drop_cols += ['percent_gender_unknown']

    df = df.drop(drop_cols, 1)

    return df


def fetch_alternative_financial_data():
    dji = pd.read_csv(YAHOO_FINANCE + "^DJI.csv")
    vix = pd.read_csv(YAHOO_FINANCE + "^VIX.csv")

    dji = dji[['Date', 'Adj Close']]
    vix = vix[['Date', 'Adj Close']]

    dji = dji.rename(columns={'Adj Close': 'dji_close'})
    vix = vix.rename(columns={'Adj Close': 'vix_close'})

    df = dji.merge(vix)

    df['year'] = df['Date'].apply(lambda x: x.split('-')[0])
    df['month'] = df['Date'].apply(lambda x: x.split('-')[1])

    df = df.drop('Date', 1)

    df = df.groupby(['year', 'month'], as_index=False).mean()
    df = df.sort_values(by=['year', 'month'])

    df['dji_sma_5'] = df['dji_close'].rolling(window=5).mean()
    df['vix_sma_5'] = df['vix_close'].rolling(window=5).mean()

    df = df.fillna(0)

    # Initial clean
    df = _strip_strings(df)
    df = _camel_case_cols(df)
    df = _remove_punctuation(df)

    return df
