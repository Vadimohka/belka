"""Ordered multiprocessing with a bounded input/result window."""
from itertools import islice


def bounded_map(pool, function, iterable, workers, chunksize=16):
    iterator = iter(iterable)
    while window := list(islice(iterator, max(1, workers) * chunksize * 2)):
        yield from pool.map(function, window, chunksize=chunksize)
