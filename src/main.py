#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# main.py
#

# There are internal structures to SKLearn and Pandas
# which throw a FutureWarning -- These are out of our control
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# === Restaurant Inspection Viewing ==================================
# import src.preprocessing.fetch as fetch
# Merged set closure rate: 3.2%
# Merging netted about 750 new closures
# df = fetch.fetch_restaurant_inspection_data(merged_set=False)
# ====================================================================

# === Restaurant Cluster Viewing =====================================
# from src.models.visualization import pca_clusters
# pca_clusters()
# ====================================================================

from src.models.prediction import ClosureRegressor

print("*** Closure Regressor ******************************")
print("** Using Master Data Set")
print("****************************************************")

c = ClosureRegressor()
c.prepare()
c.fit_gradient_boosting()
c.fit_lin_reg()
c.fit_neural_network()

print("****************************************************")

print()

print("*** Closure Classifier *****************************")
print("*** Using Restaurant Inspection Data Set ")
print("****************************************************")

c = Clos
