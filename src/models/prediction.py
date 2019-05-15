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


import pandas as pd
import sklearn.metrics as metric
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from src.preprocessing.fetch import fetch_restaurant_inspection_data
from src.preprocessing.merge import master
from src.preprocessing.transform import split_train_test, min_max_scale_values


class Model:
    """Abstract Class for the below Models"""

    def __init__(self):
        # The data set used
        self.df = None

        # The estimator that was used
        self.estimator = None

        # Name of the prediction column
        self.y_col = None

    def prepare(self):
        raise NotImplementedError

    def _fit(self):
        x_train, x_test, y_train, y_test = split_train_test(self.df.fillna(0), y=self.y_col)
        print("Training Size:", len(x_train))
        print("Test Size    :", len(x_test))

        self.estimator.fit(x_train, y_train)
        self.validate(y_test, x_test)

    def select_features(self, print_output=True, apply_and_refit=True):
        x, y, _, _ = split_train_test(self.df.fillna(0), y=self.y_col, split_size=0.0)

        selector = RFECV(self.estimator, n_jobs=-1)
        selector.fit(x, y)

        if print_output:
            print("Best Features for Current Estimator:")

            ranks = sorted(zip(selector.ranking_, self.df.columns))
            for r in ranks:
                print(r)

        if apply_and_refit:

            # Make the selected set (those ranked as 1)
            selected_set = []
            for i in range(len(selector.ranking_)):
                if selector.ranking_[i] == 1:
                    selected_set.append(self.df.columns.values[i])

            # Apply to df
            self.df = self.df[selected_set]

            # Re-fit
            self._fit()

    def validate(self, y_test, x_test):
        raise NotImplementedError


class ClosureClassifier(Model):
    """
    Performs classification of closed/not-closed
    on the Restaurant Inspections dataset. This can
    either be done via Gradient Boosting Machine or
    K-Nearest Neighbor.
    """

    def __init__(self):
        super().__init__()

        # The restaurant closure dataset
        self.df = fetch_restaurant_inspection_data()

        self.y_col = "is_closed"

    def prepare(self):
        df = fetch_restaurant_inspection_data().fillna(0)

        # Temporarily remove the Y column
        y = df[self.y_col]
        df = df.drop(self.y_col, 1)

        # Drop other unneeded columns
        drop_list = ['camis', 'dba', 'boro', 'building', 'street', 'phone',
                     'inspection_year', 'inspection_month', 'inspection_day',
                     'violation_ratio']

        df = df.drop(drop_list, 1)

        # Pivot cuisine
        cuisines = pd.get_dummies(df['cuisine_description'], prefix='cuisine')
        df = df.join(cuisines).drop('cuisine_description', 1)

        df = min_max_scale_values(df, None)

        # Place back Y col
        df[self.y_col] = y.copy()

        #
        # Downsample majority class
        #

        open_downsample = df.loc[df[self.y_col] == 0].sample(sum(df[self.y_col]) * 3,
                                                             random_state=42)

        df = pd.concat([open_downsample, df.loc[df[self.y_col] == 1]])

        self.df = df

    def fit_gradient_boosting(self):
        print("=== Gradient Boosting ========================")

        self.estimator = GradientBoostingClassifier(n_estimators=300,
                                                    learning_rate=0.3)
        self._fit()

        features_importance = sorted(zip(self.estimator.feature_importances_,
                                         self.df.columns),
                                     reverse=True)
        for f in features_importance[:10]:
            print(f)
        print("==============================================")

    def fit_knn(self):
        print("=== kNN ======================================")

        self.estimator = KNeighborsClassifier(n_neighbors=30)
        self._fit()
        print("==============================================")

    def fit_neural_network(self):
        print("=== Neural Network ===========================")
        self.estimator = MLPClassifier(hidden_layer_sizes=(20, 7, 2),
                                       alpha=0.00005,
                                       learning_rate_init=0.001,
                                       max_iter=500,
                                       random_state=11)
        self._fit()
        print("==============================================")

    def validate(self, y_test, x_test):
        print("Results:")
        print(metric.confusion_matrix(y_test, self.estimator.predict(x_test)))
        print("F1         :", metric.f1_score(y_test, self.estimator.predict(x_test)))
        print("Cohen Kappa:", metric.cohen_kappa_score(y_test, self.estimator.predict(x_test)))


class ClosureRegressor(Model):
    """
    Perform a regression of the predicted
    number of restaurant closures based upon
    the master dataset, which is arranged in
    time series per each observed zip code.
    This can either be done via a Gradient
    Boosting Machine or Linear Regression.
    """

    def __init__(self):
        super().__init__()

        # The master dataset
        self.df = master()

        self.y_col = "total_closures"

    def prepare(self):
        df = self.df.loc[pd.to_numeric(self.df['inspection_year']) != 1900]

        # Drop identifying columns
        drop_list = ['inspection_year', 'inspection_month', 'year', 'month']

        # Remove the "reason for closure" columns
        drop_list += [c for c in df.columns.values if "reason_" in c]

        print(drop_list)

        self.df = df.drop(drop_list, 1)

    def fit_gradient_boosting(self):
        print("=== Gradient Boosting ========================")

        self.estimator = GradientBoostingRegressor(n_estimators=300,
                                                   learning_rate=0.13)
        self._fit()

        features_importance = sorted(zip(self.estimator.feature_importances_,
                                         self.df.columns),
                                     reverse=True)
        for f in features_importance[:10]:
            print(f)
        print("==============================================")

    def fit_lin_reg(self):
        print("=== Linear Regression ========================")
        self.estimator = LinearRegression()
        self._fit()
        print("==============================================")

    def fit_neural_network(self):
        print("=== Neural Network ===========================")
        self.estimator = MLPRegressor(hidden_layer_sizes=(15, 5),
                                      alpha=0.001)
        self._fit()
        print("==============================================")

    def validate(self, y_test, x_test):
        print("Results:")
        print("Exp Var:", metric.explained_variance_score(y_test, self.estimator.predict(x_test)))
        print("MAE    :", metric.mean_absolute_error(y_test, self.estimator.predict(x_test)))
        print("R2     :", metric.r2_score(y_test, self.estimator.predict(x_test)))
