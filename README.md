## Vector feature selection

Mutual-information based feature selection considering feature sets rather than single-dimensional features.

The mutual information (MI) among two random variables X and Y, I(X;Y) can be computed from their joint and marginal probability density functions (pdf) *fxy*, *fx* and *fy*. The MI can be expanded to random vectors ***X*** and ***Y***, however their pdf estimation becomes much harder.

Take the three types of feature selection methods:   
  - Forward selection  
  - Backward elimination  
  - Exhaustive search  

Mutual information based selection methods in the literature traditionally follow the Forward Selection approach by using a variety of scores such as:
 - MIM
 - MRMR
 - JMIM
 - etc.

Which consist of different combinations of low-dim MI among the different candidate features and the target, keeping to the trivariate case MI(X,Y;Z) at most and avoiding the hindrance of estimating high-dimensional probability densities.

This repository provides an efficient implementation for:
  - Mutual Information MI(***X***;***Y***)
  - Forward selection methods in the literature
  - Backward elimination from the methods in the literature
  - Exhaustive selection based on the raw MI(***X***;***Y***)


##### Shorts
```
from vfs import *
from vfs.shorthands import df_iris, MRMR

df, features, targets = df_iris()

# Mutual information between two variables
mi = mi_frame(df)(['F1'], ['F2'])
print(mi)

# Mutual information between two groups of variables (vectors)
mi = mi_frame(df)(['F1','F2'], ['F3','F4', 'F5'])
print(mi)

# Select the two best features according to MRMR (forward), using shorthand
summary, __, __ = MRMR(df, ['F1', 'F2', 'F3', 'F4'], ['F5'], k=2)
print(summary)

# Select the best two features according to JMIM (backward), using default func
__, sel, disc = backward_eliminator(df, ['F1', 'F2', 'F3', 'F4'], ['F5'], k=2, loss=jmim, mi_fun=mi_frame(df))
print(sel)
print(disc)

# Select the best three features by testing all feat combinations
sel, score = exhaustive_searcher(df, ['F1', 'F2', 'F3', 'F4'], ['F5'], k=2, mi_fun=mi_frame(df))
print(sel)

# Select the best feature vector between two candidates
mifun = mi_frame(df)
aa = ['F1', 'F2']
bb = ['F3', 'F4']
best = aa if mifun(aa, ['F5']) > mifun(bb, ['F5']) else bb
```


##### Acknowledgments

This work has been partially developed within the Galicia Sur Health Research Institute (IISGS).
