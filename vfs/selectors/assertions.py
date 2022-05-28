import pandas as pd
import warnings
from ..losses import mrmr

def check_stuff(selector):
    def wrapper(df=None, features=None, targets=None, k=None, loss=None, mi_fun=None):

        # Assert types, but let them go through to raise python default exception
        assert isinstance(df, pd.DataFrame) or (not df), "df must be dataframe"
        assert isinstance(features, list) or (not features), "features must be list[str]"
        assert isinstance(targets, list) or (not targets), "targets must be list[str]"
        assert isinstance(k, int) or (not k), "k must be int"

        # Assert the number of desired features makes sense
        assert 0 < k < len(features), f"can't choose k={f} features out of nfeats={len(features)}"

        # Exceptional case: MRMR CANNOT be backwards eliminator
        if (selector.__name__ == 'backward_eliminator') and (loss is mrmr):
            warnings.warn("MRMR cannot be backwards, skipping...")
            return (None, None, None)

        return selector(df, features, targets, k, loss, mi_fun)

    return wrapper
