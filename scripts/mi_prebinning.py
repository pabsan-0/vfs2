from context import vfs
from vfs import mi_frame, mi_tensor
from vfs.shorthands import df_iris
from time import perf_counter

"""
This script compares our pandas and torch MI implementation speed with and
without prebinning. The pandas implementation should be slower. For 5 MI runs,
we experienced twice the speed by prebinning.
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
    # load some helper data
    df, features, targets = df_iris()

    # Compute MI five times, no prebinning. Pandas implementation
    with TimerStdout():
        print("\nFRAME: No prebinning, frame...\n\t", end='')
        for __ in range(5):
            print(mi_frame(df)(features, targets), end='; ')

    # Compute MI five times, with prebinning. Pandas implementation
    with TimerStdout():
        print("\nFRAME: W/ prebinning, frame...\n\t", end='')
        mifun = mi_frame(df)
        for __ in range(5):
            print(mifun(features, targets), end='; ')


    # Compute MI five times, no prebinning. Torch implementation
    with TimerStdout():
        print("\nTENSOR: No prebinning...\n\t", end='')
        for __ in range(5):
            print(mi_tensor(df, gpu=False)(features, targets), end='; ')

    # Compute MI five times, with prebinning. Torch implementation
    with TimerStdout():
        print("\nTENSOR: W/ prebinning...\n\t", end='')
        mifun = mi_tensor(df, gpu=False)
        for __ in range(5):
            print(mifun(features, targets), end='; ')
