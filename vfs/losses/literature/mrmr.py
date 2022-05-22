


class mrmr:
    """ Maximal Relevance Minimal Redundance scoring, Peng (2005). """
    def __init__(self, mi=None):
        self.mi = mi

    def choose_best(self, candidates, selected, targets, best_is_max=True):
        argtuple = (selected, targets, self.mi)
        if selected:
            try:
                ixy = candidates.feat.parallel_apply(self.bivariate_ixy, args=argtuple)
                ixw = candidates.feat.parallel_apply(self.bivar_sum_ixw, args=argtuple)
            except ValueError as e:
                print(e)
                print("@@ Forbidden: tried using MRMR for backward elimination.")

            scores = ixy - (ixw / len(selected))
        else:
            scores = candidates.feat.parallel_apply(self.bivariate_ixy, args=argtuple)

        best_score_idx = scores.idxmax() if best_is_max else scores.idxmin()
        return best_score_idx, scores[best_score_idx]

    @staticmethod
    def bivariate_ixy(candidate, __, targets, function):
        # Compute independent term I(X;Y) in MRMR loss
        return function([candidate], targets)

    @staticmethod
    def bivar_sum_ixw(candidate, selected, __, function):
        # Compute summation term sum_w{I(W;Y)} in MRMR loss
        return sum([function([candidate], [sf]) for sf in selected])
