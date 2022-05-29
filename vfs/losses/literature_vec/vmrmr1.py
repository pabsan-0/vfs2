class vmrmr1:
    """ Based on Maximal Relevance Minimal Redundance scoring, Peng (2005).
    Substitutes the estimated redundancy term 1/S*Sum_{w in s}{ I(Xw; Xc) } for
    the true JMI I({Xs}; Xc).
    """
    name = 'MRMR loss (vectorial 1)'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_ixy if first_iter else cls.mrmr_score

    @staticmethod
    def mrmr_score(cd, *args):
        return vmrmr1.bivariate_ixy(cd, *args) - vmrmr1._multivariate_ixw(cd, *args)

    @staticmethod
    def bivariate_ixy(candidate, __, targets, function):
        # Compute independent term I(X;Y) in MRMR loss
        return function([candidate], targets)

    @staticmethod
    def _multivariate_ixw(candidate, selected, __, function):
        # Compute summation term sum_w{I(W;S)} in MRMR loss
        return function([candidate], selected)
