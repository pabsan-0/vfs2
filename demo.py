import pandas as pd
from pandarallel import pandarallel

from vfs.mi.mi_base import mi_helper
from vfs.selectors import BackwardEliminator, ForwardSelector, ExhaustiveSearcher
from vfs.losses import *

pandarallel.initialize()
df = pd.read_csv('data.csv')
mifun = mi_helper(df)


losses = {
    'mim': mim,
    'disr': disr,
    'jmi': jmi,
    'jmim': jmim,
    # 'mrmr': mrmr,
    'njmim': njmim,
}


results = {}
for name, loss in losses.items():
    results[name + 'fwd'] =  ForwardSelector(df, df.columns[:-1], ['A15'],  k=3, loss=loss, mi_fun=mifun)
    results[name + 'bwd'] =  BackwardEliminator(df, df.columns[:-1], ['A15'],  k=3, loss=loss, mi_fun=mifun)
    print(results[name + 'fwd'])
    print(results[name + 'bwd'])


es = ExhaustiveSearcher(df,  df.columns[:-1], ['A15'], k=1, mi_fun=mifun)