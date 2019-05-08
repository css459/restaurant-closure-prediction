#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# visualization.py
#
# Visualization of clusters and time series data
#


import matplotlib.pyplot as plt
import numpy as np

from src.models.cluster_analysis import pca_on_restaurant_inspections_file
from src.preprocessing.merge import master
from src.preprocessing.transform import min_max_scale_values


def closures_over_time():
    """
    Returns frame with closures vs DJIA and VIX
    :return: `DataFrame`
    """
    df = master().dropna()

    # Merge Time
    df['time'] = df['inspection_year'].astype('str') + df['inspection_month'].astype('str')
    df['time'] = df['time'].astype('int')
    df = df.sort_values('time', ascending=True)

    # Isolate Cols
    df = df[['time', 'dji_close', 'vix_close', 'total_closures']]

    # Group by time
    agr = {
        'dji_close': np.mean,
        'vix_close': np.mean,
        'total_closures': np.sum
    }
    df = df.groupby('time', as_index=False).agg(agr)

    # Scale values
    df, t = min_max_scale_values(df, 'time')
    df['time'] = t

    return df.dropna()


def closures_vs_zip_code():
    """
    Graphs open and closed restaurants
    based on zip code and address
    :return: `None`
    """
    pass


def pca_clusters():
    """
    Performs PCA on the restaurant inspections
    dataset and colors points based on closed or
    not-closed
    :return: `None`
    """
    pca, colors = pca_on_restaurant_inspections_file()

    plt.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=colors)
    plt.show()
