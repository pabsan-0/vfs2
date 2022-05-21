import pandas as pd
from pandarallel import pandarallel

from vfs.mi.mi_base import mi_helper
from vfs.selectors.ForwardSelector import ForwardSelector
from vfs.losses import *

pandarallel.initialize()
df = pd.read_csv('data.csv')
mifun = mi_helper(df)



"""
mim  = ForwardSelector(df, df.columns[:-1], ['A15'],  k=10, loss=mim, mi_fun=mifun)
disr  = ForwardSelector(df, df.columns[:-1], ['A15'], k=10, loss=disr, mi_fun=mifun)
jmi   = ForwardSelector(df, df.columns[:-1], ['A15'], k=10, loss=jmi, mi_fun=mifun)
jmim  = ForwardSelector(df, df.columns[:-1], ['A15'], k=10, loss=jmim, mi_fun=mifun)
mrmr  = ForwardSelector(df, df.columns[:-1], ['A15'], k=10, loss=mrmr, mi_fun=mifun)
njmim = ForwardSelector(df, df.columns[:-1], ['A15'], k=10, loss=njmim, mi_fun=mifun)


print('mim'); print(mim)
print('disr'); print(disr)
print('jmi'); print(jmi)
print('jmim'); print(jmim)
print('mrmr'); print(mrmr)
print('njmim'); print(njmim)
"""
