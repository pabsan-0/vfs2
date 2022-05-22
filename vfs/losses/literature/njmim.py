from .. import Loss

class njmim(Loss):
    """ Joint Mutual Information Maximization, Bennasar (2015). """
    name = 'NJMIM loss'

    def __init__(self, mi=None):
        self.mi = mi

    def choose_best(self, candidates, selected, targets, best_is_max=True):
        """ Choose the best feature from 'candidates' based on this score. """
        # Define the func that will be .applied and pack its arguments
        scoring = self.jmim_score if selected else self.bivariate_mi
        argtuple = (selected, targets, self.mi)

        # Parallel computation of each feature's score & return best
        scores = candidates.feat.parallel_apply(scoring, args=argtuple)
        best_score_idx = scores.idxmax() if best_is_max else scores.idxmin()
        return best_score_idx, scores[best_score_idx]

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets, h_norm=True)

    @staticmethod
    def jmim_score(candidate, selected, targets, function):
        return min([function([candidate, sf], targets, h_norm=True) for sf in selected])
