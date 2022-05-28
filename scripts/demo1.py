import pandas as pd

from context import vfs
from vfs.mi.mi_frame import mi_frame
from vfs.selectors import backward_eliminator, forward_selector, BackwardEliminator, ForwardSelector, ExhaustiveSearcher
from vfs.selectors.exhaustive_searcher import exhaustive_searcher
from vfs.losses import *


df = pd.read_csv('scripts/_data.csv')
mifun = mi_frame(df)

es2 = exhaustive_searcher(df,  df.columns[:-1], ['A15'], k=3, mi_fun=mifun)
print(es2)
es1 = ExhaustiveSearcher(df,  df.columns[:-1], ['A15'], k=3, mi_fun=mifun)
print(es1)
