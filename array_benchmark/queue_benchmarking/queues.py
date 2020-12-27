"""Benchmark performance of put/getting numpy arrays on a single process"""

import multiprocessing as mp
from queue import Queue

import numpy as np
import timing

import pandas as pd

_TIME = timing.get_timing_group(__name__)


def baseline_benchmark(np_arr):
    """baseline"""
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array


def mp_queue_benchmark(np_arr, mp_queue):
    """multiprocessing queues"""
    mp_queue.put(np_arr)
    np_arr = mp_queue.get()
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array


def queue_module_benchmark(np_arr, queue_module):
    """queue library's queues"""
    queue_module.put(np_arr)
    np_arr = queue_module.get()
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array


def get_timings(groupname, n_frames):
    """ Get a dictionary of the mean/std and FPS of the timing group.

    :param str groupname: name of the timing group
    :param int n_frames: number of timing repeats
    :return: mean/std and FPS of the timing group as a dictionary
    :rtype: dict
    """
    mean = f"{_TIME.summary[groupname]['mean']:.4f}"
    stddev = f"{_TIME.summary[groupname]['stddev']:.4f}"
    fps = f"{n_frames / _TIME.summary[groupname]['mean']}"
    print(f"{groupname}: time: = {mean} +/- "
          f"{stddev}"
          f" or FPS = {fps}")
    return {"groupname": groupname, "mean": mean, "stddev": stddev, "fps": fps}


def benchmark_queues():
    """Benchmark various implementations of queues for put/getting numpy arrays"""
    np_arr = np.ones((1000, 1000))
    n_frames = 100
    mp_queue = mp.Queue()
    queue_module = Queue()
    timings = []

    for _ in _TIME.measure_many("baseline", samples=n_frames):
        baseline_benchmark(np_arr)

    for _ in _TIME.measure_many("mp_queue", samples=n_frames):
        mp_queue_benchmark(np_arr, mp_queue)

    for _ in _TIME.measure_many("queue_module", samples=n_frames):
        queue_module_benchmark(np_arr, queue_module)

    timings.append(get_timings("baseline", n_frames))
    timings.append(get_timings("mp_queue", n_frames))
    timings.append(get_timings("queue_module", n_frames))

    df = pd.DataFrame(timings)

    return df


if __name__ == "__main__":
    DF = benchmark_queues()
    DF.to_csv("timings/queue_timings.csv")
