class njmim:
    """ Joint Mutual Information Maximization, Bennasar (2015). """
    name = 'NJMIM loss'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_mi if first_iter else cls.njmim_score

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets, h_norm=True)

    @staticmethod
    def njmim_score(candidate, selected, targets, function):
        return min([function([candidate, sf], targets, h_norm=True) for sf in selected])
