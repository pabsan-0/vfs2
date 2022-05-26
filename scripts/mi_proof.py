import numpy as np
import sklearn.datasets
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

from context import vfs
from vfs.mi import mi_helper


"""
This script compares our MI implementation against a established one from
the sklearn library, by computing the traditional singledim I(X,Y) for X,Y
extracted from an arbitrary dataset.

Notice that sklearn functions are not deterministic, hence their results are
averaged for a number of runs.
"""



if __name__ == '__main__':

    # Load dataframe and fetch column names for arbitary feature and target
    df = sklearn.datasets.load_iris(as_frame=True).frame
    feature = df.columns[0]
    target = df.columns[-1]

    # Computing MI using scikit methods
    ff = df[feature].values.reshape(-1,1)
    yy = df[target].values
    mi_clasif = np.array([mutual_info_classif(ff, yy, n_neighbors=5) for __ in range(20)])
    mi_regres = np.array([mutual_info_regression(ff, yy, n_neighbors=5) for __ in range(20)])

    # Computing MI using our implementation
    mi_ours = mi_helper(df)([feature], [target])

    # Display, using standard deviation in %
    std  = lambda arr : round((arr.std() / (arr.std()+arr.mean())) * 100)
    print(f"""@@ Comparing our MI approach to a baseline (singledimensional):
    \rScikit classif:    {mi_clasif.mean().round(3)}  (stddev={std(mi_clasif)}%)
    \rScikit regression: {mi_regres.mean().round(3)}  (stddev={std(mi_regres)}%)
    \rOur MI:            {mi_ours}\
    """
    )
