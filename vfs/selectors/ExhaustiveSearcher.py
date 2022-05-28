import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
import queue
from math import comb
import tqdm
from ..mi.mi_frame import mi_frame


class ExhaustiveSearcher:

    def __init__(self, df, features, targets, k=3, mi_fun=None, pbar=True, memclear=True):
        # Inmutable parameters throughout the whole feature selection
        self.features: list[str] = features
        self.targets: list[str] = targets
        self.k: int = min(k, len(self.features)) if k else len(self.features)
        self.pbar = pbar

        # Heavy inmutable, should be cleaned after computations
        # Feeding INSTANCE is allowed to avoid rediscretization on repeated runs
        self.mi_function = mi_fun if mi_fun else mi_frame(df)

        # Process-specific attributes, mutable
        self.selected: list[str] = None
        self.scores: list[float] = None

        # Do the shit
        self.feature_selection_run()

        # Build summary with results
        self.summary = (self.selected, self.scores)
        self.__repr__()

        # Delete prediscretized data within mi func
        if memclear == True:
            del self.mi_function


    def __repr__(self):
        """ Instance representation with results and method information. """
        return self.summary.__repr__()


    def feature_selection_run(self):
        """ Parallel computation of the k-best features JMI-wise. Computes the
        JMI score between each combination of features & the targets. Each
        is run in a different process within a worker pool. Each worker dumps
        the result in a queue as soon as it is ready. A supervisor process runs
        in parallel and checks each value in the queue, holding on to the
        highest score it has seen. When all workers have finished, the supervisor
        flushes the queue and places the highest scoring feature subset in it.
        Beware that pool.starmap might silently raise errors.
        """
        # Multiprocessing toys for happy inter-process communication
        manager = mp.Manager()
        q_scores = manager.Queue()
        q_signal = manager.Queue()
        pool = mp.Pool()

        # Supervisor: cleans the score queue of any non-max values in real time
        supervisor = mp.Process(target=self.maxmonitor, args=(q_scores, q_signal))
        supervisor.start()

        # Define args to map function as a generator so they dont clog memory
        assert self.k > 0, 'Target number of features cannot be zero.'
        feature_combos = itertools.combinations(self.features, self.k)
        constant_args = (self.targets, self.mi_function, q_scores)
        packed_args = ((list(fc), *constant_args) for fc in feature_combos)

        # Progressbar stuff. Not reliable if too many processes but still helps
        d: str = f'Exhaustive Feature Search ({self.k} out of {len(self.features)})'
        n: int = comb(len(self.features), self.k)
        progressbar = lambda x : tqdm.tqdm(x, desc=d, total=n) if self.pbar else x

        # Have the pool compute the scores & dump them to supervised queue
        pool.starmap_async(self.scoring, progressbar(packed_args))
        pool.close()
        pool.join()

        # After workers finish, have the supervisor flush queue and terminate
        q_signal.put('flush')
        supervisor.join()
        self.selected, self.scores = q_scores.get()


    @staticmethod
    def scoring(features, targets, function, my_queue):
        """ Actual scoring function for given 'features', other args constant.
        Expensive computation meant to be applied to worker pool processes.
        May be substituted in children classes to define other methods.
        """
        score = function(features, targets)
        my_queue.put([features, score])
        return None


    @staticmethod
    def maxmonitor(my_queue, signal):
        """ Method to be applied to a process that supervises a worker pool.
        Analyzes a queue in real time, storing the highest seen value. After
        receiving a non-empty signal, flush out the queue and place the highest
        seen value in it as its only element. Typically the queue is empty when
        arriving to second phase but we flush anyway for safety. Exceptions are
        meant to catch empty queues.
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
