import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
import queue
from math import comb
import tqdm
from ..mi.mi_frame import mi_frame


def __scoring(features, targets, function, my_queue):
    """ Computes a the mutual information score of the given feature and target
    sets, then places it in the queue.
    """
    score = function(features, targets)
    my_queue.put([features, score])
    return None


def __maxmonitor(my_queue, signal):
    """ While signal queue is empty, receive elements from my_queue and keep
    only the one which has the highest [1] value. After signal is received,
    end flushing the queue and die.
    """
    # Phase 1: init + standard routine while worker pool is running
    max_value_yet = [0,0]
    while signal.empty():
        try:
            # Timeout so loop cant get stuck here & stop checking for signal
            val = my_queue.get(timeout=0.01)
            if val[1] >= max_value_yet[1]:
                max_value_yet = val
        except queue.Empty:
            continue

    # Phase 2: After getting a signal: flush queue, place maxvalue & die
    while 1:
        try:
            val = my_queue.get(timeout=0.01)
            if val[1] >= max_value_yet[1]:
                max_value_yet = val
        except queue.Empty:
            my_queue.put(max_value_yet)
            return None



def exhaustive_searcher(df, features, targets, k=3, mi_fun=None, pbar=True):

        # Inmutable parameters throughout the whole feature selection
        features: list[str] = list(features)
        targets: list[str] = list(targets)
        kk: int = min(k, len(features)) if k else len(features)

        # Progressbar stuff. Not reliable if too many processes but still helps
        d: str = f'Exhaustive Feature Search ({kk} out of {len(features)})'
        n: int = comb(len(features), k)
        progressbar = lambda x : tqdm.tqdm(x, desc=d, total=n) if pbar else x


        # Multiprocessing toys for happy inter-process communication
        manager = mp.Manager()
        q_scores = manager.Queue()
        q_signal = manager.Queue()
        pool = mp.Pool()

        # Make generator to give to workers
        assert kk > 0, 'Target number of features cannot be zero.'
        feature_combos = itertools.combinations(features, kk)
        constant_args = (targets, mi_fun, q_scores)
        packed_args = ((list(fc), *constant_args) for fc in feature_combos)

        # Supervisor: cleans the score queue of any non-max values in real time
        supervisor = mp.Process(target=__maxmonitor, args=(q_scores, q_signal))
        supervisor.start()

        # Have the pool compute the scores & dump them to supervised queue
        pool.starmap_async(__scoring, progressbar(packed_args))
        pool.close()
        pool.join()

        # After workers finish, have the supervisor flush queue and terminate
        q_signal.put('flush')
        supervisor.join()
        selected, scores = q_scores.get()

        return selected, scores
