from context import vfs
from vfs import mi_frame, mi_tensor
from vfs.shorthands import df_iris
from time import perf_counter

"""
This script compares our torch MI implementation speed on GPU and CPU, both
with and without prebinning. 
"""


class TimerStdout:
    """ Context manager to measure execution time"""
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.3f} seconds'
        print('\n\t' + self.readout)



if __name__ == '__main__':

    df, features, targets = df_iris()

    # Sometimes GPUs are slow on first contact
    # Lets instance a tensor to have that downtime outside the timed runs
    torch.Tensor([1]).to('cuda')
    
    with TimerStdout():
        print("\nTENSOR CPU: No prebinning...\n\t", end='')
        for __ in range(20):
            print(mi_tensor(df, gpu=False)(features, targets), end='; ')

    with TimerStdout():
        print("\nTENSOR CPU: W/ prebinning...\n\t", end='')
        mifun = mi_tensor(df, gpu=False)
        for __ in range(20):
            print(mifun(features, targets), end='; ')


    with TimerStdout():
        print("\nTENSOR GPU: No prebinning...\n\t", end='')
        for __ in range(20):
            print(mi_tensor(df, gpu=True)(features, targets), end='; ')

    with TimerStdout():
        print("\nTENSOR GPU: W/ prebinning...\n\t", end='')
        mifun = mi_tensor(df, gpu=True)
        for __ in range(20):
            print(mifun(features, targets), end='; ')
