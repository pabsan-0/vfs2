from context import vfs
from vfs import mi_frame, forward_selector
from vfs import vdisr, vmrmr1, vmrmr2
from vfs import disr, mrmr
from vfs.shorthands import df_iris

"""
This script runs feature selection using methods adapted from the literature
which suppress the approximation of true multidimensional MIs, and use the actual
vector MI instead.
"""

df, feats, targets = df_iris()
mi = mi_frame(df)

# DISR methods
res_disr = forward_selector(df, feats, targets, k=2, loss=disr, mi_fun=mi)
res_vdisr = forward_selector(df, feats, targets, k=2, loss=vdisr, mi_fun=mi)

# MRMR methods
res_mrmr = forward_selector(df, feats, targets, k=2, loss=mrmr, mi_fun=mi)
res_vmrmr1 = forward_selector(df, feats, targets, k=2, loss=vmrmr1, mi_fun=mi)
res_vmrmr2 = forward_selector(df, feats, targets, k=2, loss=vmrmr2, mi_fun=mi)
