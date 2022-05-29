class vmrmr2:
    """ Based on Maximal Relevance Minimal Redundance scoring, Peng (2005).
    Substitutes the redundancy term 1/S*Sum_{w in s}{ I(Xw;Y) } for the true
    JMI I({Xs}; Y), and the first term I(Xc,Y) for the relevance of the whole
    selected set I({Xs+Xc}, Y).
    """
    name = 'MRMR loss (vectorial 2)'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_ixy if first_iter else cls.mrmr_score

    @staticmethod
    def mrmr_score(cd, *args):
        return vmrmr2._multivariate_ixy(cd, *args) - vmrmr2._multivariate_ixw(cd, *args)

    @staticmethod
    def bivariate_ixy(candidate, __, targets, function):
        # Compute independent term I(X;Y) in MRMR loss
        return function([candidate], targets)

    @staticmethod
    def _multivariate_ixy(candidate, selected, targets, function):
        # Compute independent term I(X;Y) in MRMR loss
        return function([*selected, candidate], targets)

    @staticmethod
    def _multivariate_ixw(candidate, selected, __, function):
        # Compute summation term sum_w{I(W;Y)} in MRMR loss
        return function([candidate], selected)
