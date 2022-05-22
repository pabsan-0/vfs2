import numpy as np
import pandas as pd
from ..mi.mi_base import mi_helper


class BackwardEliminator:

    def __init__(self, df, features, targets, k=3, loss=None, mi_fun=None, memclear=True):
        # Inmutable parameters throughout the whole feature selection
        self.features: list[str] = list(features)
        self.targets: list[str] = list(targets)
        self.k: int = min(k, len(self.features)) if k else 1

        # Heavy inmutable, should be cleaned after computations
        # Feeding INSTANCE is allowed to avoid rediscretization on repeated runs
        self.loss = loss(mi_fun if mi_fun else mi_helper(df))

        # Process-specific attributes, mutable
        self.candidates: DataFrame = self.list_to_frame(self.features)
        self.discarded: list[str] = []
        self.scores: list[float] = []

        # Do the shit
        self.feature_selection_run()

        # Build summary with results
        data = np.array([self.discarded, self.scores]).T
        self.summary = pd.DataFrame(data, columns=['Discarded', 'Score'])
        self.__repr__()

        # Delete prediscretized data within mi func
        if memclear == True:
            del self.loss.mi
            del self.candidates


    def __repr__(self):
        """ Instance representation with results and method information. """
        return self.summary.__repr__()


    def feature_selection_run(self):
        # Discard until we have the right number of features.
        while len(self.candidates) > self.k:
            selected = list(self.features.copy())
            [selected.remove(ii) for ii in self.discarded]

            # Inspect all features and choose the best one
            feat, score = self.loss.choose_best(self.candidates, selected, self.targets,
                best_is_max=False)

            # Arrange stacks
            self.candidates.drop(feat, inplace=True)
            self.discarded.append(feat)
            self.scores.append(score)


    @staticmethod
    def list_to_frame(my_list) -> pd.DataFrame:
        """ Used for the Candidates stack, to parallelize with pandas."""
        my_frame = pd.DataFrame({'feat': my_list})
        my_frame = my_frame.set_index(['feat'], drop=False)
        return my_frame
