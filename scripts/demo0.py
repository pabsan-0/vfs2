import pandas as pd

from context import vfs
from vfs.mi.mi_frame import mi_frame
from vfs.selectors import backward_eliminator, forward_selector, BackwardEliminator, ForwardSelector, ExhaustiveSearcher
from vfs.losses import *


df = pd.read_csv('scripts/_data.csv')
mifun = mi_frame(df)


losses = {
    'mim': mim,
    'disr': disr,
    'jmi': jmi,
    'jmim': jmim,
    'mrmr': mrmr,
    'njmim': njmim,
}


results = {}
for name, loss in losses.items():
    y =  ForwardSelector(df, df.columns[:-1], ['A15'],  k=3, loss=loss, mi_fun=mifun)
    z =  forward_selector(df, df.columns[:-1], ['A15'],  k=10, loss=loss, mi_fun=mifun)
    print(name)
    #print(results[name + 'fwd'])
    print(y)
    print(z)


es = ExhaustiveSearcher(df,  df.columns[:-1], ['A15'], k=1, mi_fun=mifun)
