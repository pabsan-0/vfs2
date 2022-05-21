## Vector feature selection

Mutual-information based feature selection considering feature sets rather than single-dimensional features.

Three types of feature selection methods:   
  - Forward selection   
  - Backward elimination
  - Exhaustive search

Any of the above depends on a loss function. Mutual information based selection methods use various metrics as loss function. Some losses available in the literature are:
 - MIM
 - MRMR
 - JMIM
 - etc.

Any of the above loss functions depend on the way mutual information (MI) is computed. So far, MI is computed for pairs of features in all methods available in the literature. MI requires estimating their joint density functions of the considered features, which is complex for more than a few. We propose two implementations of a cheap multi-feature mutual information estimator.
  - Partial histograms: a straightforward evolution of the current MI approach
  - Sample based: alternative approach faster the less available samples

These selectors may be run either on CPU or a GPU.
