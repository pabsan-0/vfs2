import numpy as np
import pandas as pd
from ..mi.mi_frame import mi_frame
from .assertions import check_stuff

@check_stuff
def backward_eliminator(df, features, targets, k=3, loss=None, mi_fun=None):

    # Inmutable parameters throughout the whole feature selection
    features: list[str] = list(features)
    targets: list[str] = list(targets)
    kk: int = min(k, len(features)) if bool(k) else 1

    # Process-specific attributes, mutable
    candidates = pd.DataFrame({'feat': features}).set_index(['feat'], drop=False)
    selected = list(features.copy())
    discarded: list[str] = []
    scores: list[float] = []

    while len(candidates) > kk:
        # Pack arguments and loss to send to multiprocessing
        _args = (selected, targets, mi_fun)
        _loss = loss.choose(first_iter=not selected)

        # Multiprocessing and choose the best feature this iter
        iter_scores = candidates.feat.parallel_apply(_loss, args=_args)
        feat  = iter_scores.idxmin()
        score = iter_scores[feat]

        # Manage selected/discarded/etc
        candidates.drop(feat, inplace=True)
        discarded.append(feat)
        selected.remove(feat)
        scores.append(score)

    # Build summary dataframe with the ranking
    data = np.array([discarded, scores]).T
    summary = pd.DataFrame(data, columns=['Discarded', loss.name])

    assert (selected + discarded).sort() == features.sort()
    return summary, selected, discarded
