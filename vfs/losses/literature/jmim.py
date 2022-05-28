class jmim:
    """ Joint Mutual Information Maximization, Bennasar (2015)."""
    name = 'JMIM loss'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_mi if first_iter else cls.jmim_score

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets, h_norm=False)

    @staticmethod
    def jmim_score(candidate, selected, targets, function):
        return min([function([candidate, sf], targets, h_norm=False) for sf in selected])
