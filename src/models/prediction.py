#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# prediction.py
#
# File to generate regression estimation of restaurant closures
# Prediction also for restaurant closure classification for restaurant
# inspection dataset
#

from src.preprocessing.fetch import fetch_restaurant_inspection_data
from src.preprocessing.merge import master


class ClosureClassifier:
    """
    Performs classification of closed/not-closed
    on the Restaurant Inspections dataset. This can
    either be done via Gradient Boosting Machine or
    K-Nearest Neighbor.
    """

    def __init__(self):
        # The restaurant closure dataset
        df = fetch_restaurant_inspection_data()

        # The estimator that was used
        # (either GBM or kNN)
        estimator = None

    def prepare(self):
        pass

    def fit_gradient_boosting(self):
        pass

    def fit_knn(self):
        pass

    def validate(self):
        pass


class ClosureRegressor:
    """
    Perform a regression of the predicted
    number of restaurant closures based upon
    the master dataset, which is arranged in
    time series per each observed zip code.
    This can either be done via a Gradient
    Boosting Machine or Linear Regression.
    """

    def __init__(self):
        # The master dataset
        df = master()

        # The estimator that was used
        # (either GBM or Linear Regression)
        estimator = None

    def prepare(self):
        pass

    def fit_gradient_boosting(self):
        pass

    def fit_lin_reg(self):
        pass

    def validate(self):
        pass
