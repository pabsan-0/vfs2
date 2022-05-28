class mrmr:
    """ Double Input Symmetrical Relevance, Meyer (2006). """
    name = 'DISR loss'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_ixy if first_iter else cls.mrmr_score

    @staticmethod
    def mrmr_score(cd, *args):
        return mrmr.bivariate_ixy(cd, *args) - mrmr._bivar_sum_ixw(cd, *args)

    @staticmethod
    def bivariate_ixy(candidate, __, targets, function):
        # Compute independent term I(X;Y) in MRMR loss
        return function([candidate], targets)

    @staticmethod
    def _bivar_sum_ixw(candidate, selected, __, function):
        # Compute summation term sum_w{I(W;Y)} in MRMR loss
        return sum([function([candidate], [sf]) for sf in selected]) / len(selected)
