import pandas as pd
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

"""
This script has a few wrapper functions that call R scripts running the praznik
library to get different results towards feature selection with mutual
information. See Kursa 2021.
"""

def mi_score(csv: str, feature: str, target: str) -> float:
    """ Praznik library R wapper.
    Computes the mutual information between two features in a dataset loaded
    from 'csv'. Specify the feature and target names as string.
    """
    score = robjects.r(f"""\
        library('praznik')
        df <- read.csv('{csv}')
        X <- df${feature}
        Y <- df${target}
        miScores(X,Y)
        """)
    return str(score).split()[-1]


def feat_selection(csv: str, fun='JMI', k=3, n_digits=5) -> [list, list]:
    """ Praznik library R wapper.
    Performs feature selection from the data loaded in 'csv' according to
    a method specified with 'fun'. Selects a total of 'k' features and rounds
    the result to 'n_digits.'
    """
    feats, scores = robjects.r(f"""\
        library('praznik')
        df <- read.csv('{csv}')
        X <- subset(df, select=-c(Class))
        Y <- df$Class

        {fun}(X, Y, {k})
        """)
    feats = [name for name, index in feats.items()]
    scores = [round(i, n_digits) for i in scores]
    return feats, scores


if __name__ == '__main__':

    # Base variables loading. Load csv in python just to get variable names.
    filename = 'data/thyroid-dis.csv'
    df = pd.read_csv(filename)
    features = df.columns.to_list()[:-1]
    targets = df.columns.to_list()[-1:]

    # Showcasing standalone MIs
    print('Standalone I(Xi;Y)')
    print('Feature  Score')
    for feature in features:
        a = mi_score(filename, feature, 'Class')
        print(feature, '\t', a)

    # Showcasing the MI
    fs_kursa, scores_kursa = feat_selection(csv=filename, fun='JMI', k=10)
    print('\nKursas JMI feature selection:')
    print('Feature\t\tScore\n', *[f'{a}\t\t{b}\n' for a,b in zip(fs_kursa, scores_kursa)])
