
class jmi:
    """ (Pairwise) Joint Mutual Information scoring, Yang (1999). """
    def __init__(self, mi=None):
        self.mi = mi

    def choose_best(self, candidates, selected, targets, best_is_max=True):
        # Define the func that will be .applied and pack its arguments
        scoring = self.trivariate_mi if selected else self.bivariate_mi
        argtuple = (selected, targets, self.mi)

        # Parallel computation of each feature's score & return best
        scores = candidates.feat.parallel_apply(scoring, args=argtuple)
        best_score_idx = scores.idxmax() if best_is_max else scores.idxmin()
        return best_score_idx, scores[best_score_idx]

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets)

    @staticmethod
    def trivariate_mi(candidate, selected, targets, function):
        return sum([function([candidate, sf], targets) for sf in selected])
