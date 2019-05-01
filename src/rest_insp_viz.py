#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from src.fetch import fetch_restaurant_inspection_data

#
# Prepare Initial Clean
#

ids = ['camis', 'dba', 'boro', 'building', 'street', 'phone',
       'inspection_year', 'inspection_month', 'inspection_day']

df = fetch_restaurant_inspection_data()
df = df.loc[pd.to_numeric(df['inspection_year']) != 1900]

df = df.drop(ids, 1)
df = df.dropna()

#
# Encode Cuisine Label
#

from sklearn.preprocessing import LabelEncoder

cuisine_labels = LabelEncoder().fit_transform(df['cuisine_description'])
df = df.drop('cuisine_description', 1)

#
# Extract the is_closed labels
#

is_closed_labels = np.where(df['is_closed'], 1, 0)
df = df.drop('is_closed', 1)
df['closed'] = is_closed_labels

#
# Downsample majority class
#

open_downsample = df.loc[df['closed'] == 0].sample(sum(is_closed_labels) * 3, random_state=42)
s = pd.concat([open_downsample, df.loc[df['closed'] == 1]])

#
# Re-upsample with replacement on Sub-sample set
#

# df_resample = resample(s, n_samples=2000, replace=False, random_state=42)

df_resample = s
is_closed_labels_resample = df_resample['closed']
df_resample = df_resample.drop('closed', 1)

# from sklearn.cluster import SpectralClustering
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df_resample, is_closed_labels_resample)
#
# sc = SpectralClustering(n_clusters=2, n_neighbors=100, n_jobs=-1)
# # sc = SpectralClustering(affinity='rbf', n_jobs=-1)
#
# y_pred = sc.fit_predict(df_resample, is_closed_labels_resample)
#
#
#
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(is_closed_labels_resample, y_pred))
#
# from sklearn.metrics import f1_score
# print(f1_score(is_closed_labels_resample, y_pred))

# exit(0)

#
# Perform Dimensionality Reduction
#

from sklearn.decomposition import PCA

p = PCA(n_components=3)
pca = p.fit_transform(df_resample)

#
# Plot the Reduced Set
#

from matplotlib import pyplot as plt

plt.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=is_closed_labels_resample, alpha=0.3)
plt.show()

#
# Fit a GBM on Reduced Dimensions
#

from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.5)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_resample, is_closed_labels_resample)

print(list(df_resample.columns))

p = PCA(n_components=3)
# X_train = p.fit_transform(X_train)
# X_test = p.transform(X_test)

gbm.fit(X_train, y_train)

#
# Evaluate
#

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, gbm.predict(X_test)))

from sklearn.metrics import f1_score

print(f1_score(y_test, gbm.predict(X_test)))

#
# Test Trained Classifier on Entire Dataset
#

# Initial Prep
is_closed_labels = df['closed']
df = df.drop('closed', 1)

# PCA
from sklearn.decomposition import PCA

p = PCA(n_components=3)
pca = p.fit_transform(df)

# Visualization
from matplotlib import pyplot as plt

plt.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=is_closed_labels, alpha=0.3)
plt.show()

# from sklearn.ensemble import GradientBoostingClassifier
# gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.2)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, is_closed_labels)

# X_train = p.transform(X_train)
# X_test = p.transform(X_test)

# gbm.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, gbm.predict(X_test)))

from sklearn.metrics import f1_score

print(f1_score(y_test, gbm.predict(X_test)))

#
# # In[22]:
#
#
# dfdd['closed'] = is_closed_labels
# violations_closed = dfdd.loc[dfdd['closed'] == 1]['violation_count']
#
#
# # In[23]:
#
#
# violations_open = dfdd.loc[dfdd['closed'] == 0]['violation_count']
#
#
# # In[30]:
#
#
# from sklearn.preprocessing import MinMaxScaler
# violations_closed = MinMaxScaler().fit_transform(np.array(violations_closed).reshape(-1, 1))
# violations_open = MinMaxScaler().fit_transform(np.array(violations_open).reshape(-1, 1))
#
#
# # In[32]:
#
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# from matplotlib import pyplot
# x = violations_open
# y =violations_closed
#
# bins = np.linspace(-1.0, 1.0, 100)
#
# pyplot.hist(x, bins, alpha=0.5, label='closed')
# pyplot.hist(y, bins, alpha=0.5, label='open')
# pyplot.legend(loc='upper right')
# pyplot.show()
#
#
# # In[25]:
#
#
# dfdd.corr()


# In[ ]:
