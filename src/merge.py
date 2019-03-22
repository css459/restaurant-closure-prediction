#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# merge.py
#

# This file handles the operations from which data files
# are merged together into larger files. For data matrix
# creation, see `preprocessing.py`

import os

import pandas as pd

import src.fetch as fetch

#
# Constants
#


MERGED_FILE_PATH = fetch.DATA_DIR + 'merged/'


#
# Helper Functions
#


def _total_restaurant_closure(restaurant_dataset):
    r = restaurant_dataset[['is_closed', 'inspection_year', 'inspection_month', 'zip']]
    r = r.loc[r['is_closed']]
    return r.groupby(['inspection_year', 'inspection_month', 'zip'], as_index=False).count()


def _violation_distribution_table(restaurant_dataset):
    r = restaurant_dataset[['is_closed', 'inspection_year',
                            'inspection_month', 'zip', 'violation_code']]

    # First, generate a list of violations to pivot on
    # This list is generated from using the top 15 violations from closed and not-closed
    # restaurants and taking only the values which do not appear in the non-closed set
    l1 = r.loc[r['is_closed']]['violation_code'].value_counts(sort=True)[:15]
    l2 = r.loc[~r['is_closed']]['violation_code'].value_counts(sort=True)[:15]
    violations = ["violation_" + c for c in l1.index if c not in l2.index]

    # Pivot Violations
    dummies = pd.get_dummies(r['violation_code'], prefix='violation')
    dummies = dummies[violations]
    r = r.join(dummies).drop('violation_code', 1)

    r = r.loc[r['is_closed']].drop('is_closed', 1)

    return r.groupby(['inspection_year', 'inspection_month', 'zip'], as_index=False).sum()


def _cuisine_distribution_table(restaurant_dataset):
    r = restaurant_dataset[['is_closed', 'inspection_year',
                            'inspection_month', 'zip', 'cuisine_description']]

    r = r.loc[r['is_closed']].drop('is_closed', 1)

    r = r.join(pd.get_dummies(r['cuisine_description'],
                              prefix='cuisine')).drop('cuisine_description', 1)

    return r.groupby(['inspection_year', 'inspection_month', 'zip'], as_index=False).sum()


def _inspection_table():
    i = fetch.fetch_inspection_data()
    i = i[['inspection_year', 'inspection_month',
           'zip', 'reason', 'is_closed']]

    # i = i.loc[i['is_closed'] == 1].drop('is_closed', 1)
    i = i.join(pd.get_dummies(i['reason'], prefix='reason')).drop('reason', 1)
    i = i.groupby(['inspection_year', 'inspection_month', 'zip'], as_index=False).sum()

    i = i.rename(columns={'is_closed': 'total_closures'})

    print(i['total_closures'].sum())

    return i.drop('reason_na', 1)


#
# Table Merging Functions
#


def closure_data(reload=False):
    """
    The closure dataset is a merging of the Inspection
    and Restaurant inspection datasets. This is done
    at the zip code and monthly level where each row
    is a closure. The follow columns are:
        total_closures: The total number of hard closures
        violation_*:    Percent of closures with that type of violation
                        based on soft closures
        cuisine_*:      Percent of closures with that type of cuisine served
                        based on soft closures
        reason_*:       Total occurrences of closure reasons

    :param reload:  Force rewrite of cached CSV file

    :return:        DataFrame
    """
    file_name = MERGED_FILE_PATH + "closures.csv"

    if not reload and os.path.isfile(file_name):
        return pd.read_csv(file_name)

    r1 = fetch.fetch_restaurant_inspection_data(include_violation_code=True)
    r2 = fetch.fetch_restaurant_inspection_data()

    r_closures = _total_restaurant_closure(r2)
    r_violations = _violation_distribution_table(r1)
    r_cuisines = _cuisine_distribution_table(r2)

    r_merged_monthly = pd.merge(r_closures, r_violations,
                                on=['inspection_year', 'inspection_month', 'zip'])

    r_merged_monthly = pd.merge(r_merged_monthly, r_cuisines,
                                on=['inspection_year', 'inspection_month', 'zip'])

    inspections = _inspection_table()

    full = pd.merge(r_merged_monthly, inspections,
                    on=['inspection_year', 'inspection_month', 'zip'],
                    how='outer')

    # Make violation and cuisine columns percent distributions based upon
    # the soft closures from the restaurant inspections. Then, this number
    # can be extrapolated to represent the total hard closure (total_closures)

    cols = [c for c in full.columns if "violation_" in c or "cuisine_" in c]

    for c in cols:
        full[c] = full[c] / full['is_closed']

    for c in [cc for cc in full.columns if "reason_" in cc]:
        full[c] = full[c] / full['total_closures']

    # Cleanup

    full = full.fillna(0)
    full['total_closures'] = full['total_closures'] + full['is_closed']
    full = full.drop('is_closed', 1)
    full = full.sort_values(['inspection_year', 'inspection_month', 'zip'], ascending=False)

    full = fetch._strip_strings(full)
    full = fetch._camel_case_cols(full)
    full = fetch._remove_punctuation(full)

    full.to_csv(file_name, index=False)

    return full


# Merged set of demographic information by zip
# code. Each cell represents a percent of the whole
# for that row
def demographic_data():
    taxes = fetch.fetch_alternative_agi_returns()
    demo = fetch.fetch_alternative_demographic_stats_data()

    return taxes.merge(demo)


# Economic data with SMA/EMA per day
def economic_data():
    # This is the same as the financial data in
    # fetch.py for now
    return fetch.fetch_alternative_financial_data()


# TODO
# Important keyword frequencies, rating, and count of reviews
# for each found restaurant (row).
def reviews_data():
    pass


# Entirely merged table where possible
def master():
    pass
