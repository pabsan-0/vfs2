import numpy as np
import pandas as pd
import torch
from .mi_frame import mi_frame



class mi_tensor(mi_frame):
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
