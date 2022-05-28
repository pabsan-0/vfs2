from context import vfs
from vfs import mi_frame, backward_eliminator, forward_selector, exhaustive_searcher
from vfs import mim, disr, jmi, jmim, mrmr, njmim
from vfs.shorthands import df_iris

"""
This script is meant to provide a quick overview on how to use this library by
running less feature selection methods and taking a moment to preview what
comes out of the functions.
"""

if __name__ == '__main__':

    # Load dataframe and prebin it
    df, features, targets = df_iris()
    mifun = mi_frame(df)

    print("The variable 'mifun' is a callable, not an MI.")
    print("It will make repeated MIs faster by prebinning our features. ")
    print(mifun)



    summ, sel, disc = forward_selector(df, features, targets,  k=3, loss=jmim, mi_fun=mifun)
    print('\nForward Selection:')
    print('A summary of the selection ranking:')
    print(summ)
    print('Forward selection chose these features:', sel)
    print('but it did not choose these:', disc)

    summ, sel, disc = backward_eliminator(df, features, targets,  k=3, loss=jmim, mi_fun=mifun)
    print('\nBackward elimnation:')
    print('A summary of the discarding ranking:')
    print(summ)
    print('Backward elimnation did not discard these:', sel)
    print('These are the ones it discarded:', disc)

    feats, score = exhaustive_searcher(df, features, targets, k=2, mi_fun=mifun)
    print('\nExhaustive search::')
    print('This is different. This algorithm tested all feature combinations.')
    print('This is the feature set found optimal:', feats)
    print('And this was the score it got:', score)
