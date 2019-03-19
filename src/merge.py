#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# merge.py
#

# This file handles the operations from which data files
# are merged together into larger files. For data matrix
# creation, see `preprocessing.py`

import src.fetch as fetch


# Merged set where each row is a restaurant
# with closure indicator and record date
def closure_data():
    pass


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
