import pandas as pd
from pandarallel import pandarallel

from context import vfs
from vfs.mi.mi_frame import mi_frame
from vfs.selectors import BackwardEliminator, ForwardSelector, ExhaustiveSearcher
from vfs.losses import *

pandarallel.initialize()
df = pd.read_csv('scripts/_data.csv')
mifun = mi_frame(df)


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
