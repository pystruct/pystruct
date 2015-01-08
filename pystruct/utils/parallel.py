import warnings
import sys
import signal
from functools import wraps
from multiprocessing.dummy import Pool as ThreadPool
try:
    from sklearn.externals.joblib.pool import (MemmapingPool, Pool) 
except ImportError:
    from multiprocessing import Pool
    MemmapingPool = False 
    warnings.warn("your scikit-learn version does not include "
                  "MemmapingPool, all parallelization using "
                  "multiprocessing.Pool")
from sklearn.externals.joblib import cpu_count



## decorators for handling 
## exceptions and logging in worker processes
class KeyboardInterruptError(Exception): pass


def handle_exceptions(func):
    """handle exceptions in worker pool more gracefully"""
    @wraps(func)
    def func_handle_exceptions(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise KeyboardInterruptError()
    return func_handle_exceptions


def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)


class ParallelMixin(object):
    """mixin class for all parallelization"""

    def _spawn_pool(self):
        """spawn pool attribute."""
        if self.n_jobs == -1:
            self._n_jobs = cpu_count()
        else:
            self._n_jobs = self.n_jobs
        if any([self.n_jobs == 1, self.use_threads]):
            self.pool = ThreadPool(processes=self._n_jobs)
        elif all([self.use_memmapping_pool, MemmapingPool]):
            self.pool = MemmapingPool(processes=self._n_jobs, 
                    initializer=init_worker,
                    temp_folder=self.memmapping_temp_folder)
        else:
            self.pool = Pool(processes=self._n_jobs, 
                    initializer=init_worker)


    def __getstate__(self):
        """strip pool when pickling"""
        odict = self.__dict__.copy()
        del odict['pool']
        return odict


    def __setstate__(self, idict):
        """spawn pool when unpickling"""
        self.__dict__ = idict
        self._spawn_pool()

    
    def parallel(self, func, args_iterable, timeout=sys.maxint):
        if self.pool is None:
            self._spawn_pool()
        results_async = self.pool.map_async(func, args_iterable)
        while 1:
            try:
                if results_async.ready():
                    results = results_async.get(sys.maxint)
                    return results
                else:
                    pass
            except KeyboardInterrupt:
                raise KeyboardInterruptError()




