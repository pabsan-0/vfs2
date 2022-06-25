from context import vfs
from vfs import mi_frame, backward_eliminator, forward_selector, exhaustive_searcher
from vfs import mim, disr, jmi, jmim, mrmr, njmim
from vfs.shorthands import df_iris

"""
This script is meant to provide a quick overview on how to use this library by
running most implemented feature selection methods.
"""

if __name__ == '__main__':

    # Load dataframe and prebin it
    df, features, targets = df_iris()
    mifun = mi_frame(df)

    # Run all Forward selections
    for loss in [mim, disr, jmi, jmim, mrmr, njmim]:
        a =  forward_selector(df, features, targets,  k=3, loss=loss, mi_fun=mifun)
        print('Forward ' + loss.name)
        print(a[0])
        print(a[1:], end='\n\n')

    # Run all Backward selections
    for loss in [mim, disr, jmi, jmim, mrmr, njmim]:
        a =  backward_eliminator(df, features, targets,  k=3, loss=loss, mi_fun=mifun)
        print('Backward ' + loss.name)
        print(a[0])
        print(a[1:], end='\n\n')

    # Run Exhaustive search
    es = exhaustive_searcher(df, features, targets, k=3, mi_fun=mifun)
    print("Exhaustive\n", es)
