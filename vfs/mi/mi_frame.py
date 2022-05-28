import numpy as np
import pandas as pd
from .onehot import onehot_enable


class mi_frame:
    """ Feature selection helper for faster repeated computations. Instancing
    this class will return a callable object with prediscretized data, so
    successive MI calls are significantly faster.

    # Example 1
    mi_fun = mi_helper(df, n_bins, n_digits)
    mi_val = mi_fun(features, targets)

    # Example 2
    mi_val = mi_helper(df, n_bins, n_digits)(features, targets)
    """
    def __init__(self, df: pd.DataFrame, n_bins=10, n_digits=3):
        """ Loads base arguments and prebins data for later use.
        """
        # Base hyper params
        self.n_bins = n_bins
        self.n_digits = n_digits

        # Infer some basics and prebin the data
        self.cols = df.columns.to_list()
        self.n_samples = len(df)
        self.binned = self.sample_prebinning(df)


    @onehot_enable
    def __call__(self, feat_x: list, feat_y: list, h_norm=False) -> float:
        """ Base instance operation. Checks healthy inputs and then runs MI.
        """
        # Input sanity checks
        assert isinstance(feat_x, list), f"Computing MI: Provided 'feat_x' not list: {feat_x}."
        assert isinstance(feat_y, list), f"Computing MI: Provided 'feat_y' not list: {feat_y}."
        assert isinstance(h_norm, bool), f"Computing MI: Provided 'h_norm' not bool: {h_norm}."

        # Compute mutual information acc to samplebased one method
        mi, pxy = self.samplebased_mutualinfo(feat_x, feat_y, h_norm)

        # Mehod sanity checks
        assert .97 < pxy.sum() < 1.03, f'Probability space not covered: pxy.sum() = {pxy.sum()}.'
        assert mi >= 0, f'Mutual information yields negative value: {mi}'

        # Return solely mutual information
        return mi


    def sample_prebinning(self, df):
        """ Discretize each feature and put each sample into a bin.
        """
        binned = df.copy()
        for __, feat in enumerate(self.cols):
            binned[feat] = \
                pd.cut(binned[feat],
                       self.n_bins,
                       precision=self.n_digits,
                       right=False,
                       labels=False,
                       include_lowest=True)
        return binned


    def samplebased_mutualinfo(self, feat_x: list, feat_y: list, h_norm=False):
        """ Computes the standard mutual information across two random vectors
        X and Y, I(X;Y). If 'h_norm' returns I(X;Y)/H(*X,*Y) instead.
        """
        # Sample-based-estimate pdfs, compute base MI & entropy-normalize if so
        bb  = self.pdf_estimation(self.binned, feat_x, feat_y)
        mi  = self.mutualinformation(bb)
        mi /= self.mutualentropy(bb) if h_norm else 1

        # Return final value. Rounding was made before binning, but insist here
        return mi.round(self.n_digits), bb.pxy


    @staticmethod
    def pdf_estimation(binned: pd.DataFrame, feat_x: list, feat_y: list):
        """ Sample-based multivariable probability distribution estimator.
        Estimates the probability of occurrence of each single sample from
        a discretized dataframe, by instance counting and then dividing by the
        total number of samples. Computes marginal and joint probabilities for
        the features in the sets X and Y. Returns dataframe with data + probs.
        """
        # Generate disposable copy of binned df, sliced to only the considered
        # features + targets so that the duplicate removal trick works out later.
        featxy = [*feat_x, *feat_y]
        bb = binned.copy()[featxy]

        # Count repeated instances and divide by n_samples=len(binned)
        # groupby()[0] is REQUIRED so duplicates dont vanish.
        bb['pxy'] = bb.groupby(featxy)[featxy[0]].transform('count') / len(binned)
        bb['px']  = bb.groupby(feat_x)[featxy[0]].transform('count') / len(binned)
        bb['py']  = bb.groupby(feat_y)[featxy[0]].transform('count') / len(binned)

        # After each sample has its probability as attribute, drop duplicates
        return bb.drop_duplicates()


    @staticmethod
    def mutualinformation(bb: pd.DataFrame):
        """ Computes the mutual information of a set of RV given their joint
        and marginal probability distributions as feats in the input dataframe.
        """
        bb['Ixy_integral_bit'] = bb.pxy * np.log(bb.pxy / bb.px / bb.py)
        mi = bb['Ixy_integral_bit'].sum()
        return mi


    @staticmethod
    def mutualentropy(bb: pd.DataFrame):
        """ Computes the mutual entropy of a set of RV given their joint
        probability distribution as feats in the input dataframe.
        """
        bb['Hxy_integral_bit'] = -bb.pxy * np.log(bb.pxy)
        h = bb['Hxy_integral_bit'].sum()
        return h
