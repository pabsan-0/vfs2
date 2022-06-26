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

This repository provides implementations for:
  - Mutual Information MI(***X***;***Y***)
  - Forward selection methods in the literature
  - Backward elimination from the methods in the literature
  - Exhaustive selection based on the raw MI(***X***;***Y***)

:boom: See next section [Shorts](#shorts) for a super brief tutorial on how to use the library.  

:boom: See subdirectory [scripts](scripts) for more detailed examples.



### Shorts
```
from vfs import *
from vfs.shorthands import df_iris, MRMR
df, features, targets = df_iris()
```
##### Mutual information
```
# Mutual information between two variables
mi = mi_frame(df)(['F1'], ['F2'])
print(mi)

# Mutual information between two groups of variables (vectors)
mi = mi_frame(df)(['F1','F2'], ['F3','F4', 'F5'])
print(mi)
```


##### Traditional feature selection
```
# Select the two best features according to MRMR (forward), using shorthand
summary, __, __ = MRMR(df, ['F1', 'F2', 'F3', 'F4'], ['F5'], k=2)
print(summary)

# Select the best two features according to JMIM (backward), using default func
__, sel, disc = backward_eliminator(df, ['F1', 'F2', 'F3', 'F4'], ['F5'], k=2, loss=jmim, mi_fun=mi_frame(df))
print(sel)
print(disc)
```

##### Vectorial feature selection
```
# Select the best three features by testing all feat combinations
sel, score = exhaustive_searcher(df, ['F1', 'F2', 'F3', 'F4'], ['F5'], k=2, mi_fun=mi_frame(df))
print(sel)

# Select the best feature vector between two candidates
mifun = mi_frame(df)
aa = ['F1', 'F2']
bb = ['F3', 'F4']
best = aa if mifun(aa, ['F5']) > mifun(bb, ['F5']) else bb
```



### Bibliography

- <div id="ref-battiti1994" class="csl-entry" role="doc-biblioentry"> Battiti, Roberto. 1994. <span>“Using Mutual Information for Selecting Features in Supervised Neural Net Learning.”</span> <em>IEEE Transactions on Neural Networks</em> 5 (4): 537–50. </div><br/>

- <div id="ref-yang1999" class="csl-entry" role="doc-biblioentry">Yang, H, and John Moody. 1999. <span>“Feature Selection Based on Joint Mutual Information.”</span> In <em>Proceedings of International ICSC Symposium on Advances in Intelligent Data Analysis</em>, 1999:22–25. Citeseer. </div><br/>

- <div id="ref-fleuret2004" class="csl-entry" role="doc-biblioentry"> Fleuret, François. 2004. <span>“Fast Binary Feature Selection with Conditional Mutual Information.”</span> <em>Journal of Machine Learning Research</em> 5 (9). </div><br/> 

- <div id="ref-peng2005" class="csl-entry" role="doc-biblioentry"> Peng, Hanchuan, Fuhui Long, and Chris Ding. 2005. <span>“Feature Selection Based on Mutual Information Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy.”</span> <em>IEEE Transactions on Pattern Analysis and Machine Intelligence</em> 27 (8): 1226–38. </div><br/>

- <div id="ref-meyer2006" class="csl-entry" role="doc-biblioentry"> Meyer, Patrick E, and Gianluca Bontempi. 2006. <span>“On the Use of Variable Complementarity for Feature Selection in Cancer Classification.”</span> In <em>Workshops on Applications of Evolutionary Computation</em>, 91–102. Springer. </div><br/>

- <div id="ref-bennasar2015" class="csl-entry" role="doc-biblioentry"> Bennasar, Mohamed, Yulia Hicks, and Rossitza Setchi. 2015. <span>“Feature Selection Using Joint Mutual Information Maximisation.”</span> <em>Expert Systems with Applications</em> 42 (22): 8520–32. </div> <br/>

- <div id="ref-bommert2020" class="csl-entry" role="doc-biblioentry"> Bommert, Andrea, Xudong Sun, Bernd Bischl, Jörg Rahnenführer, and Michel Lang. 2020. <span>“Benchmark for Filter Methods for Feature Selection in High-Dimensional Classification Data.”</span> <em>Computational Statistics &amp; Data Analysis</em> 143: 106839. </div><br/>

- <div id="ref-kursa2021" class="csl-entry" role="doc-biblioentry"> Kursa, Miron B. 2021. <span>“Praznik: High Performance Information-Based Feature Selection.”</span> <em>SoftwareX</em> 16: 100819. </div><br/>

