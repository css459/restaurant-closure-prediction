#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# main.py
#


import warnings

from src.models.prediction import ClosureRegressor, ClosureClassifier
from src.models.visualization import pca_clusters

# There are internal structures to SKLearn and Pandas
# which throw a FutureWarning -- These are out of our control
warnings.simplefilter(action='ignore', category=FutureWarning)

# === Restaurant Inspection Viewing ==================================
# import src.preprocessing.fetch as fetch
# Merged set closure rate: 3.2%
# Merging netted about 750 new closures
# df = fetch.fetch_restaurant_inspection_data(merged_set=False)
# ====================================================================

# === Restaurant Cluster Viewing =====================================
pca_clusters()
# ====================================================================

# === Closure Prediction =============================================
print("*** Closure Regressor ******************************")
print("*** Using Master Data Set")
print("****************************************************")

c = ClosureRegressor()
c.prepare()
c.fit_gradient_boosting()
c.select_features()
c.fit_lin_reg()
c.fit_neural_network()

print("****************************************************")

print()

print("*** Closure Classifier *****************************")
print("*** Using Restaurant Inspection Data Set ")
print("****************************************************")

c = ClosureClassifier()
c.prepare()
c.fit_gradient_boosting()

# This fails on the re-fit, if you run it, you'll
# still get the feature list
# c.select_features()

c.fit_knn()
c.fit_neural_network()
# ====================================================================
