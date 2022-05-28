class jmi:
    """ (Pairwise) Joint Mutual Information scoring, Yang (1999). """
    name = 'JMI loss'

    @classmethod
    def choose(cls, first_iter=False):
        return cls.bivariate_mi if first_iter else cls.trivariate_mi

    @staticmethod
    def bivariate_mi(candidate, __, targets, function):
        return function([candidate], targets)

    @staticmethod
    def trivariate_mi(candidate, selected, targets, function):
        return sum([function([candidate, sf], targets) for sf in selected])
