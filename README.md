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
from vfs.shorthands import df_iris

df, features, targets = df_iris()

# Mutual information between two variables
mi = mi_frame(df)(features[:0], features[-1:])
print(mi)

# Mutual information between two groups of variables (vectors)
mi = mi_frame(df)(features, targets)
print(mi)

# Select the best three features according to MRMR (forward)
fs = MRMR(df, features, targets, k=3)
print(fs)

# Select the best three features according to JMIM (backward)
be = BE
print(be)

# Select the best three features by testing all combinations
es = ES
print(es)

# Select the best feature group between two candidates

```



##### Acknowledgments

This work has been partially developed within the Galicia Sur Health Research Institute (IISGS).
