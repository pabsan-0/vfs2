from abc import ABC, abstractmethod
from vfs.mi.mi_base import mi_helper


class Loss(ABC):
    """ Parent loss function class. Feature selection losses can be different
    depending on the number of features that HAVE already been selected. Hence
    our loss functions are actually classes which pack functions.

    We have decided the simpler way to implement MI-based losses is to define a
    choose_best method to call additional static methods with actual losses.
    """

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def choose_best(candidates, selected, targets, best_is_max=True):
        pass



class __example(Loss):
    name = 'Example'

    def __init__(self, mi=None):
        self.mi: mi_helper = mi

    def choose_best(self, candidates, selected, targets, best_is_max=True):
        """ Choose the best feature from candidates to add to the selected set.
        Ranks features according to a scoring method. The best feature may be set
        to be the once that achieves either the max or min score.
        """
        # Define the func that will be .applied and pack its arguments
        scoring = self.jmim_score if selected else self.bivariate_mi
        argtuple = (selected, targets, self.mi)

        # Parallel computation of each feature's score & return best
        scores = candidates.feat.parallel_apply(scoring, args=argtuple)
        best_score_idx = scores.idxmax() if best_is_max else scores.idxmin()
        return best_score_idx, scores[best_score_idx]

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets, h_norm=False)

    @staticmethod
    def jmim_score(candidate, selected, targets, function):
        return min([function([candidate, sf], targets, h_norm=False) for sf in selected])


# this should Not return an error
__example()
