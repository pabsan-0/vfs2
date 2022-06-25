# Scripts

The scripts in this directory are supposed to serve as a little introduction to the library, as well as to showcase its main contributions.

### Contents
- [Mutual information](#mutual-information)
  - [mi_baseline.py](#page_facing_up-mi_baselinepy): comparing our MI against scikit's 
  - [mi_prebinning.py](#page_facing_up-mi_prebinningpy): prebinning data to save time when computing MI
  - [mi_gpu.py](#page_facing_up-mi_gpupy): time benchmarking CPU vs GPU
  - [mi_onehot.py](#page_facing_up-mi_onehotpy): how to use onehot-encoded data
- [Feature selection](#feature-selection)
  - [fs_showcase.py](#page_facing_up-fs_showcasepy): how to call all feature selection methods
  - [fs_verbose.py](#page_facing_up-fs_verbosepy): insights on feture selection python objects and return values
  - [fs_vectorial.py](#page_facing_up-fs_vectorialpy): demo on adapted literature methods to use true joint MI 

## Mutual information


#### [:page_facing_up: mi_baseline.py](mi_baseline.py)
>- This script compares our MI implementation with sklearn's as a sanity check.
>- Notice that sklearn's implementation does not output multivariate MI results.
>- If fed a multivariate `XX`:
>  - Scikit will return the tuple `[I(i; Y) for i in XX]` .
>  - Our implementation will return the single value `I(XX; Y) = I([i for i in XX]; Y)` .
>
> <p><details><summary> See output </summary>
>
>```
>$ python3 mi_baseline.py
>@@ Comparing our MI approach to a baseline (singledimensional):
>Scikit classif:    0.509  (stddev=2%)
>Scikit regression: 0.506  (stddev=2%)
>Our MI:            0.502   
>```
>
></details></p>


#### [:page_facing_up: mi_prebinning.py](mi_prebinning.py)
>  - Prebinning the data initially can greatly speed up multiple successive MI computations later. This is:
>    - We first discretize the whole dataset `df` into an object we call `mifun`.
>    - Any subsequential MI among any of the variables in `df` will be way faster since data is prebinned.
>    - This massively boosts speed of feature selection algorithms, which repeteadly compute MI combinations.
>  - The script will compute MI five times in a row with both our pytorch and pandas implementations, both with and without prebinning.
>  - It will print out a time benchmarking of these four scenarios.
>
><details><summary> See output </summary><p>
>
>```
>$ python3 mi_prebinning.py
>FRAME: No prebinning, frame...
>	1.089; 1.089; 1.089; 1.089; 1.089;
>	Time: 0.181 seconds
>
>FRAME: W/ prebinning, frame...
>	1.089; 1.089; 1.089; 1.089; 1.089;
>	Time: 0.132 seconds
>
>TENSOR: No prebinning...
>	1.089; 1.089; 1.089; 1.089; 1.089;
>	Time: 0.105 seconds
>
>TENSOR: W/ prebinning...
>	1.089; 1.089; 1.089; 1.089; 1.089;
>	Time: 0.051 seconds
>```
>
></p></details>  


#### [:page_facing_up: mi_gpu.py](mi_gpu.py)
>  - This script runs our pytorch implementation both on GPU and CPU, and both with and without prebinning.
>  - It will print out a time benchmarking of these four scenarios.
><details><summary> No output available </summary><p></p></details>
	

#### [:page_facing_up: mi_onehot.py](mi_onehot.py)
>  - This example shows how to compute mutual information with with one-hot encoded variables.
>  - For equally-discretized data, it shows that `I(X;Y) == I(onehot(X); Y)`
>
><details><summary> See output </summary>
><p>
>
>```
>$ python3 mi_onehot.py
>One hot encoded MI matches categorical representation, as expected:
>Categorical frame:      1.089
>Categorical tensor:     1.089
>One-hot frame:          1.089
>One-hot tensor:         1.089
>```
>
></p>
></details>


## Feature selection

#### [:page_facing_up: fs_showcase.py](fs_showcase.py):
>  - Spawns some example data from the iris dataset.
>  - Create an efficient function to compute faster mutual informations for these data (see example mi_prebinning.py).
>  - All MI-based FS literature methods are used to select the 3 best features according to the forward and backward selection paradigms.
>  - The true joint MI of all combinations of features is used to, again, take the best 3.
>
><details><summary> See output </summary>
><p>
>	
>```
>$ python3 fs_showcase.py
>Forward MIM loss
>  Selected MIM loss
>0       F4    0.979
>1       F3    0.936
>2       F1    0.502
>(['F4', 'F3', 'F1'], ['F2'])
>
>Forward DISR loss
>  Selected DISR loss
>0       F4     0.465
>1       F3     0.351
>2       F1     0.595
>(['F4', 'F3', 'F1'], ['F2'])
>
>Forward JMI loss
>  Selected JMI loss
>0       F4    0.979
>1       F1    1.034
>2       F3    2.024
>(['F4', 'F1', 'F3'], ['F2'])
>
>Forward JMIM loss
>  Selected JMIM loss
>0       F4     0.979
>1       F1     1.034
>2       F3     0.992
>(['F4', 'F1', 'F3'], ['F2'])
>
>Forward MRMR loss
>  Selected             MRMR loss
>0       F4                 0.979
>1       F3   -0.1449999999999999
>2       F2  -0.21600000000000003
>(['F4', 'F3', 'F2'], ['F1'])
>
>Forward NJMIM loss
>  Selected NJMIM loss
>0       F4      0.465
>1       F3      0.351
>2       F1      0.297
>(['F4', 'F3', 'F1'], ['F2'])
>
>Backward MIM loss
>  Discarded MIM loss
>0        F2    0.302
>(['F1', 'F3', 'F4'], ['F2'])
>
>Backward DISR loss
>  Discarded DISR loss
>0        F2     0.881
>(['F1', 'F3', 'F4'], ['F2'])
>
>Backward JMI loss
>  Discarded            JMI loss
>0        F2  3.1159999999999997
>(['F1', 'F3', 'F4'], ['F2'])
>
>Backward JMIM loss
>  Discarded JMIM loss
>0        F2     0.302
>(['F1', 'F3', 'F4'], ['F2'])
>
>vfs/selectors/assertions.py:19: UserWarning: MRMR cannot be backwards, skipping...
>Backward MRMR loss
>None
>(None, None)
>
>Backward NJMIM loss
>  Discarded NJMIM loss
>0        F2      0.109
>(['F1', 'F3', 'F4'], ['F2'])
>
>Exhaustive Feature Search (3 out of 4): 100%|██████████████████████████████████████████| 4/4 [00:00<00:00, 239.15it/s]
>Exhaustive
> (1.073, ['F2', 'F3', 'F4'], [None, None, None])
>
>```
>													   
></p>
></details>


#### [:page_facing_up: fs_verbose.py](fs_verbose.py):
>  - This script focuses on giving you a deeper overview on the python objects taken and returned from the functions in the previous example.  
>  - Basically, it runs a single feature selection run of each type: forward, backward and exhaustive.
>
><details><summary> See output </summary>
><p>
>
>```
>$ python3 fs_verbose.py
>The variable 'mifun' is a callable, not an MI.
>It will make repeated MIs faster by prebinning our features.
><vfs.mi.mi_frame.mi_frame object at 0x7f90e632d070>
>
>Forward Selection:
>A summary of the selection ranking:
>  Selected JMIM loss
>0       F4     0.979
>1       F1     1.034
>2       F3     0.992
>Forward selection chose these features: ['F4', 'F1', 'F3']
>but it did not choose these: ['F2']
>
>Backward elimnation:
>A summary of the discarding ranking:
>  Discarded JMIM loss
>0        F2     0.302
>Backward elimnation did not discard these: ['F1', 'F3', 'F4']
>These are the ones it discarded: ['F2']
>Exhaustive Feature Search (3 out of 4): 100%|██████████████████████████████████████████| 4/4 [00:00<00:00, 545.87it/s]
>
>Exhaustive search::
>This is different. This algorithm tested all feature combinations.
>This is the feature set found optimal: ['F2', 'F3', 'F4']
>And this was the score it got: 1.073
>
>Lets compare the joint MI score of the selected features for all of the above:
>Forward selected:    ['F4', 'F1', 'F3'], with joint MI = 1.067
>Backward selected:   ['F1', 'F3', 'F4'], with joint MI = 1.067
>Exhaustive selected: ['F2', 'F3', 'F4'], with joint MI = 1.073
>```
>													   
></p>
></details>

#### [:page_facing_up: fs_vectorial.py](fs_vectorial.py)
>  - This script holds true joint mutual information adaptions for the literature methods MRMR and DISR.
>  - Originally, these estimated multivariate MIs by averaging many bi/trivariate ones.
>  - Our MI computation allows changing these estimates for true multivariate mutual informations.
><details><summary> No output available </summary><p></p></details>


