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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score
from sklearn.neural_network import MLPRegressor

from src.preprocessing.fetch import fetch_restaurant_inspection_data
from src.preprocessing.merge import master
from src.preprocessing.transform import split_train_test


class Model:

    def __init__(self):
        # The data set used
        self.df = None

        # The estimator that was used
        self.estimator = None

    def prepare(self):
        raise NotImplementedError

    def _fit(self):
        x_train, x_test, y_train, y_test = split_train_test(self.df.fillna(0))
        print("Training Size:", len(x_train))
        print("Test Size    :", len(x_test))

        self.estimator.fit(x_train, y_train)
        self.validate(y_test, x_test)

    def select_features(self, estimator):
        pass

    def validate(self, y_test, x_test):
        raise NotImplementedError


class ClosureClassifier:
    """
    Performs classification of closed/not-closed
    on the Restaurant Inspections dataset. This can
    either be done via Gradient Boosting Machine or
    K-Nearest Neighbor.
    """

    def __init__(self):
        # The restaurant closure dataset
        self.df = fetch_restaurant_inspection_data()

    def prepare(self):
        pass

    def fit_gradient_boosting(self):
        pass

    def fit_knn(self):
        pass

    def fit_neural_network(self):
        pass

    def validate(self):
        pass


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
        # The master dataset
        super().__init__()

        self.df = master()

        # # The estimator that was used
        # # (either GBM or Linear Regression)
        # self.estimator = None

    def prepare(self):
        # Name of the column to predict upon
        y_col = "total_closures"

        df = self.df.loc[pd.to_numeric(self.df['inspection_year']) != 1900]

        # Drop identifying columns
        drop_list = ['inspection_year', 'inspection_month', 'zip', 'year', 'month']

        # Remove the "reason for closure" columns
        drop_list += [c for c in df.columns.values if "reason_" in c]

        print(drop_list)

        self.df = df.drop(drop_list, 1)

        # Pivot cuisine
        # cuisines = pd.get_dummies(df['cuisine_description'], prefix='cuisine')
        # df = df.join(cuisines).drop('cuisine_description', 1)

        # df = min_max_scale_values(df, None)

        # is_closed_labels = np.where(df[y_col], 1, 0)

        #
        # Downsample majority class
        #

        # open_downsample = df.loc[df[y_col] == 0].sample(sum(is_closed_labels) * 3,
        #                                                       random_state=42)
        #
        # s = pd.concat([open_downsample, df.loc[df[y_col] == 1]])

        #
        # Re-upsample with replacement on Sub-sample set
        #

        # df_resample = resample(s, n_samples=2000, replace=False, random_state=42)

        # df_resample = s
        # is_closed_labels_resample = df_resample[y_col]
        # df_resample = df_resample.drop(y_col, 1)

    def fit_gradient_boosting(self):
        print("=== Gradient Boosting ========================")

        self.estimator = GradientBoostingRegressor(n_estimators=200,
                                                   learning_rate=0.1)
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

    def select_features(self, estimator):
        pass

    def validate(self, y_test, x_test):
        print("Results:")
        print("Exp Var:", explained_variance_score(y_test, self.estimator.predict(x_test)))
        print("MAE    :", mean_absolute_error(y_test, self.estimator.predict(x_test)))
        print("R2     :", r2_score(y_test, self.estimator.predict(x_test)))
