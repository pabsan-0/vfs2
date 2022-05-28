from .losses import *
from .mi import *
from .selectors import *


FS = forward_selector
BE = backward_eliminator
ES = exhaustive_searcher


def MIM(df, features, targets, k=3, mi_fun=None):
    mifun = mi_fun if mi_fun else mi_frame(df)
    return forward_selector(df, features, targets,  k, loss=mim, mi_fun=mifun)


def DISR(df, features, targets, k=3, mi_fun=None):
    mifun = mi_fun if mi_fun else mi_frame(df)
    return forward_selector(df, features, targets,  k, loss=disr, mi_fun=mifun)


def JMI(df, features, targets, k=3, mi_fun=None):
    mifun = mi_fun if mi_fun else mi_frame(df)
    return forward_selector(df, features, targets,  k, loss=jmi, mi_fun=mifun)


def JMIM(df, features, targets, k=3, mi_fun=None):
    mifun = mi_fun if mi_fun else mi_frame(df)
    return forward_selector(df, features, targets,  k, loss=jmim, mi_fun=mifun)


def MRMR(df, features, targets, k=3, mi_fun=None):
    mifun = mi_fun if mi_fun else mi_frame(df)
    return forward_selector(df, features, targets,  k, loss=mrmr, mi_fun=mifun)


def NJMIM(df, features, targets, k=3, mi_fun=None):
    mifun = mi_fun if mi_fun else mi_frame(df)
    return forward_selector(df, features, targets,  k, loss=njmim, mi_fun=mifun)


def df_iris(rename=True):
    import sklearn.datasets
    df = sklearn.datasets.load_iris(as_frame=True).frame
    if rename:
        df.columns = ['F1', 'F2', 'F3', 'F4', 'F5']
    features = df.columns.to_list()[:-1]
    targets = df.columns.to_list()[-1:]
    return df, features, targets


__all__ = [MIM, DISR, JMI, JMIM, MRMR, NJMIM, df_iris, FS, BE, ES]
