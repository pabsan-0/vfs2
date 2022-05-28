import numpy as np
import pandas as pd
from ..mi.mi_frame import mi_frame


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

    return selected, discarded, summary







class ForwardSelector:

    def __init__(self, df, features, targets, k=3, loss=None, mi_fun=None, memclear=True):

        # Inmutable parameters throughout the whole feature selection
        self.features: list[str] = features
        self.targets: list[str] = targets
        self.k: int = min(k, len(self.features)) if k else len(self.features)

        # Heavy inmutable, should be cleaned after computations
        # Feeding INSTANCE is allowed to avoid rediscretization on repeated runs
        self.loss = loss
        self.mi = mi_fun

        # Process-specific attributes, mutable
        self.candidates: DataFrame = self.list_to_frame(self.features)
        self.selected: list[str] = []
        self.scores: list[float] = []

        # Do the shit
        self.feature_selection_run()

        # Build summary with results
        data = np.array([self.selected, self.scores]).T
        self.summary = pd.DataFrame(data, columns=['Selected', 'Score'])
        self.__repr__()

        # Delete prediscretized data within mi func
        if memclear == True:
            del self.loss
            del self.candidates


    def __repr__(self):
        """ Instance representation with results and method information. """
        return self.summary.__repr__()


    def feature_selection_run(self):

        # Select as many features as desired
        while len(self.selected) < self.k:

            # Parallel computation of each feature's score & return best
            args = lambda : (self.selected, self.targets, self.mi)
            loss = self.loss.choose(first_iter=not self.selected)
            scores = self.candidates.feat.parallel_apply(loss, args=args())
            feat = scores.idxmax()
            score = scores[feat]

            # Arrange stacks
            self.candidates.drop(feat, inplace=True)
            self.selected.append(feat)
            self.scores.append(score)


    @staticmethod
    def list_to_frame(my_list) -> pd.DataFrame:
        """ Used for the Candidates stack, to parallelize with pandas."""
        my_frame = pd.DataFrame({'feat': my_list})
        my_frame = my_frame.set_index(['feat'], drop=False)
        return my_frame
