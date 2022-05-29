class vdisr:
    """ Based on Double Input Symmetrical Relevance, Meyer (2006). """
    name = 'DISR loss, vectorial'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_mi if first_iter else cls.disr_score

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets, h_norm=True)

    @staticmethod
    def disr_score(candidate, selected, targets, function):
        return function([candidate, *selected], targets, h_norm=True)
