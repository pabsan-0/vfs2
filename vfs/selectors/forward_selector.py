import numpy as np
import pandas as pd
from ..mi.mi_frame import mi_frame
from .assertions import check_stuff

@check_stuff
def forward_selector(df, features, targets, k=3, loss=None, mi_fun=None):

    # Inmutable parameters throughout the whole feature selection
    features: list[str] = list(features)
    targets: list[str] = list(targets)
    kk: int = min(k, len(features)) if k else 1

    # Process-specific attributes, mutable
    candidates = pd.DataFrame({'feat': features}).set_index(['feat'], drop=False)
    selected: list[str] = []
    scores: list[float] = []

    while len(selected) < kk:

        # Pack arguments and loss to send to multiprocessing
        _args = (selected, targets, mi_fun)
        _loss = loss.choose(first_iter=not selected)

        # Multiprocessing and choose the best feature this iter
        iter_scores = candidates.feat.parallel_apply(_loss, args=_args)
        feat  = iter_scores.idxmax()
        score = iter_scores[feat]

        # Manage selected/discarded/etc
        candidates.drop(feat, inplace=True)
        selected.append(feat)
        scores.append(score)

    # Build summary dataframe with the ranking + discarded set
    data = np.array([selected, scores]).T
    summary = pd.DataFrame(data, columns=['Selected', loss.name])
    discarded = features; [discarded.remove(i) for i in selected]

    return summary, selected, discarded
