from .. import Loss

class disr(Loss):
    """ Double Input Symmetrical Relevance, Meyer (2006). """
    name = 'DISR loss'

    def __init__(self, mi=None):
        self.mi = mi

    def choose_best(self, candidates, selected, targets, best_is_max=True):
        # Define the func that will be .applied and pack its arguments
        scoring = self.disr_score if selected else self.bivariate_mi
        argtuple = (selected, targets, self.mi)

        # Parallel computation of each feature's score & return best
        scores = candidates.feat.parallel_apply(scoring, args=argtuple)
        best_score_idx = scores.idxmax() if best_is_max else scores.idxmin()
        return best_score_idx, scores[best_score_idx]

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets, h_norm=True)

    @staticmethod
    def disr_score(candidate, selected, targets, function):
        return sum([function([candidate, sf], targets, h_norm=True) for sf in selected])
