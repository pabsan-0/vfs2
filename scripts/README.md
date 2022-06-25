# Scripts

The scripts in this directory are supposed to serve as a little introduction to the library, as well as showcasing its main contributions.


## Mutual information

#### [mi_baseline.py](mi_baseline.py)
  - This script compares our MI implementation with sklearn's as a sanity check.
  - Notice that sklearn's implementation does not output multivariate MI results.
  - If fed a multivariate `XX`:
    - Scikit will return the tuple `[I(i; Y) for i in XX]` .
    - Our implementation will return the single value `I(XX; Y) = I([i for i in XX]; Y)` .
#### [mi_prebinning.py](mi_prebinning.py)

#### [mi_gpu.py](mi_gpu.py)
  - This script runs 
  - 
#### [mi_onehot.py](mi_onehot.py)
  - This example shows how to compute mutual information with with one-hot encoded variables.
  - It also proves that `I(X;Y) == I(onehot(X); Y)`

## Feature selection
These scripts showcase basic feature selection commands.

#### [quickstart_1.py](quickstart_1.py):
  - Spawns some example data from the iris dataset. 
  - Create an efficient function to compute faster mutual informations for these data (see example mi_prebinning.py). 
  - All MI-based FS literature methods are used to select the 3 best features according to the forward and backward selection paradigms.
  - The true joint MI of all combinations of features is used to, again, take the best 3. 
 
#### [quickstart_2.py](quickstart_2.py):
  - This script focuses on giving you a deeper overview on the python objects taken and returned from the functions in the previous example.  
  - Basically, it runs a single feature selection run of each type: forward, backward and exhaustive.

#### [fs_vectorial.py](fs_vectorial.py)
  - This script holds true joint mutual information adaptions for the literature methods MRMR and DISR.
  - Originally, these estimated multivariate MIs by averaging many bi/trivariate ones. 
  - Our MI computation allows changing these estimates for true multivariate mutual informations.





pypraznik.py
