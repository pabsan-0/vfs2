import numpy as np
import torch
import pandas as pd
from line_profiler import LineProfiler

"""
This script implements mutual information base methods. Find in this script a
CPU and a GPU implementation, plus an input capture decorator to allow one hot
feature encoding.
"""



def onehot_enable(method):
    """ Patch for the __call__ method inside mi_function to allow one-hot
    encoded features. Chooses the onehot encoded children features from a list
    with the names of their parents, relying on the separator character '#'.
        > Expected parents name: [A1]            <- this is user input
        > Expected children names: [A1#a, A1#b]. <- this is already in the class
    """
    def wrapper(instance, feat_x, feat_y, **kwargs):

        # Get all children onehot features from their parent names
        feat_x = [col for col in instance.cols if col.split('#')[0] in feat_x]
        feat_y = [col for col in instance.cols if col.split('#')[0] in feat_y]

        # Proceed as usual with the children instead of the parents.
        value = method(instance, feat_x, feat_y, **kwargs)
        return value
    return wrapper



class mi_helper:
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





class mi_helper_gpu(mi_helper):
    """ MI GPU implementation on torch, forked off the cpu pandas method. This
    class substitutes inherits the init + prebinning + call and substitutes the
    other methods. See parent class for documentation.
    """
    def __init__(self, *args, gpu=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Boot gpu if arg + allowed
        use_gpu = gpu and torch.cuda.is_available()
        self.device = 'cuda' if use_gpu else 'cpu'

    def samplebased_mutualinfo(self, feat_x: list, feat_y: list, h_norm=False):
        Px, Py, Pxy = self.pdf_estimation(self.binned, feat_x, feat_y, self.device)
        mi  = self.mutualinformation(Px, Py, Pxy)
        mi /= self.mutualentropy(Pxy) if h_norm else 1
        return mi.cpu().numpy().round(self.n_digits), Pxy

    @staticmethod
    def pdf_estimation(binned, feat_x: list, feat_y: list, device):
        # Columns to tensor, in the right order + get numeric separator + nsamples
        Txy = torch.tensor(binned[feat_x + feat_y].values).to(device)
        s = len(feat_x)
        ns = len(binned)

        # This to save horizontal space
        kwargs = {'return_counts': True, 'return_inverse': True, 'dim': 0}

        # Estimate marginal and joint probabilities for each sample
        __, indices, values = torch.unique(Txy[:,:s], **kwargs)
        Px = torch.index_select(values, 0, indices) / ns
        __, indices, values = torch.unique(Txy[:,s:], **kwargs)
        Py = torch.index_select(values, 0, indices) / ns
        __, indices, values = torch.unique(Txy[:,:], **kwargs)
        Pxy = torch.index_select(values, 0, indices) / ns

        # Merge each probability with its sample and remove duplicates
        Pxy = torch.cat([Txy.T, *[ii.unsqueeze(dim=1).T for ii in [Px, Py, Pxy]]])
        Pxy = Pxy.unique(dim=1)

        # Return the probability vectors Px, Py, Pxy
        return Pxy[-3], Pxy[-2], Pxy[-1]

    @staticmethod
    def mutualinformation(Px, Py, Pxy):
        return ( Pxy * torch.log(Pxy / Px / Py) ).sum()

    @staticmethod
    def mutualentropy(Pxy):
        return ( - Pxy * torch.log(Pxy) ).sum()





if __name__ == '__main__':
    # Custom module loading for tests
    from dataset_utils import _credit as dataset
    from dataset_utils import Dataset
    df, __, features, targets, __ = Dataset(**dataset).unpack()

    # For proper time benchmarking, start cuda outside the timer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy = torch.Tensor([[1]]).to(device)
    del dummy

    # Actual test run of both CPU and GPU mutual information
    profiler = LineProfiler()
    @profiler
    def function_to_compare_times():
        mi_fun_gpu = mi_helper_gpu(df, 10, 3)
        mi_gpu = mi_fun_gpu(features, targets)

        mi_fun = mi_helper(df, 10, 3)
        mi_cpu = mi_fun(features, targets)
        return mi_gpu, mi_cpu

    mi_gpu, mi_cpu = function_to_compare_times()
    print(f'Value check: device={device}')
    print('GPU:', mi_gpu)
    print('CPU:', mi_cpu)
    print()
    profiler.print_stats()
