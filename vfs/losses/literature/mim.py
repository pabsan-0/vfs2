class mim:
    """ Mutual Information Maximization, Battiti (1994), forward selection. """
    name = 'MIM loss'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_mi

    @staticmethod
    def bivariate_mi(candidate, __, targets, mi):
        return mi([candidate], targets)
