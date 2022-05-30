from context import vfs
from vfs import mi_frame, mi_tensor
from vfs.shorthands import df_iris
import numpy as np
import pandas as pd

"""
This script tests MI implementations with and without one hot encoded data.
According to the math, they should (and do) return equal MI.
"""

if __name__ == '__main__':
    # Load iris dataset
    df, features, targets = df_iris()

    # Lets add an artificial categorical feature with 4 unique values
    df['C1'] = np.random.randint(0, 4, len(df))
    assert len(df.C1.unique()) == 4

    # Make two dataframes: one w/ categorical and other w/ one-hot encoded variable
    df_vanilla = df
    df_onehot = pd.get_dummies(df, columns=['C1'], prefix_sep='#')

    # Compute mis for categorical data
    mi_vanilla_frame = mi_frame(df_vanilla)(features + ['C1'], targets)
    mi_vanilla_tensor = mi_tensor(df_vanilla, gpu=False)(features + ['C1'], targets)

    # Compute mis for onehot encoded data
    mi_onehot_frame = mi_frame(df_onehot)(features + ['C1'], targets)
    mi_onehot_tensor = mi_tensor(df_onehot, gpu=False)(features + ['C1'], targets)

    print('One hot encoded MI matches categorical representation, as expected:')
    print('Categorical frame:\t', mi_vanilla_frame)
    print('Categorical tensor:\t', mi_vanilla_tensor)
    print('One-hot frame:\t', mi_onehot_frame)
    print('One-hot tensor:\t', mi_onehot_tensor)
