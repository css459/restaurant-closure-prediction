#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# prediction.py
#
# Cluster generation routines for latent restaurant segmentation
#


from sklearn.decomposition import PCA

from src.preprocessing.fetch import fetch_restaurant_inspection_data
from src.preprocessing.transform import min_max_scale_values


def pca_on_master_file(n=3):
    """
    Performs a PCA decomposition on the
    master dataset (as defined in merge.py)

    :param n: The number of dimensions to keep
    :return:  `Numpy Matrix, Labels Array`
    """
    pass


def pca_on_restaurant_inspections_file(n=3, y='is_closed'):
    """
    Performs a PCA decomposition on the
    restaurant inspections dataset (as defined in
    fetch.py)

    :param n: The number of dimensions to keep
    :param y: The name of the prediction col (to color)
    :return:  `Numpy Matrix, Labels Array`
    """
    print("[ INF ] Beginning Restaurant Violation PCA")
    print("[ INF ] Fetching...")
    df = fetch_restaurant_inspection_data().dropna()
    colors = df[y]

    df = df.drop(y, 1)

    # Drop other unneeded columns
    drop_list = ['camis', 'dba', 'boro', 'building', 'street', 'zip', 'phone',
                 'inspection_year', 'inspection_month', 'inspection_day', 'violation_ratio', 'cuisine_description']

    df = df.drop(drop_list, 1)

    # Pivot cuisine
    # cuisines = pd.get_dummies(df['cuisine_description'], prefix='cuisine')
    # df = df.join(cuisines).drop('cuisine_description', 1)

    df = min_max_scale_values(df, None)

    # Create PCA representation
    print("[ INF ] Beginning PCA Decomposition: N =", n)

    pca = PCA(n_components=n).fit_transform(df)

    print("[ INF ] Done.")

    return pca, colors
