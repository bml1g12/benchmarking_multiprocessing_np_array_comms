"""Benchmark performance of put/getting numpy arrays on a single process"""

import multiprocessing as mp
import threading
import time
from queue import Queue

import numpy as np
import pandas as pd
import timing

_TIME = timing.get_timing_group(__name__)


def baseline_benchmark(np_arr_shape):
    """baseline"""
    np_arr = np.random.random(np_arr_shape)  # produce a fresh array
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array


def queue_1thread_module(np_arr_shape, queue_module):
    """queue library's queue. All is done on one thread here, and so it seems Python is smart
    enough not to serialise/pickle the data."""
    np_arr = np.random.random(np_arr_shape)  # produce a fresh array
    queue_module.put(np_arr)
    np_arr = queue_module.get()
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array


def queue_multithread_module(np_arr_shape, queue, n_frames):
    """Passing the numpy array from a producer thread to a consumer thread via a queue.
    The pickling makes this extremely slow."""
    thread = threading.Thread(target=worker_producer,
                              args=(np_arr_shape, queue, n_frames),
                              daemon=True)
    thread.start()
    time1 = time.time()
    consumer(n_frames, queue)  # will consume n_frames from producer
    time2 = time.time()
    return time2 - time1


def mp_queue_1proc_benchmark(np_arr_shape, mp_queue):
    """multiprocessing queues. All is done on one process here, and so it seems Python is smart
    enough not to serialise/pickle the data."""
    np_arr = np.random.random(np_arr_shape)  # produce a fresh array
    mp_queue.put(np_arr)
    np_arr = mp_queue.get()
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array


def mp_queue_multiproc_benchmark(np_arr_shape, queue, n_frames):
    """Passing the numpy array from a producer process to a consumer process via a queue.
    The pickling makes this extremely slow."""
    proc = mp.Process(target=worker_producer,
                      args=(np_arr_shape, queue, n_frames))
    proc.start()
    time1 = time.time()
    consumer(n_frames, queue)  # will consume n_frames from producer
    time2 = time.time()
    proc.terminate()
    return time2 - time1


def worker_producer(np_arr_shape, queue, n_frames):
    """A frame producer function, e.g. for a worker thread or process"""
    for _ in range(n_frames):
        np_arr = np.random.random(np_arr_shape)  # produce a fresh array
        queue.put(np_arr)


def consumer(n_frames, queue):
    """A frame consumer function, which draws frames from the worker thread/process via a queue
    and does a dummy calculation on the result."""
    for _ in range(n_frames):
        np_arr = queue.get()
        _ = np_arr * 2  # example of some processing done on the array


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
    np_arr_shape = (1000, 1000)
    n_frames = 100
    mp_queue = mp.Queue()
    queue_module = Queue()
    timings = []

    for _ in _TIME.measure_many("baseline", samples=n_frames, threshold=3):
        baseline_benchmark(np_arr_shape)

    for _ in _TIME.measure_many("queue_module", samples=n_frames, threshold=3):
        queue_1thread_module(np_arr_shape, queue_module)

    for _ in _TIME.measure_many("mp_queue", samples=n_frames, threshold=3):
        mp_queue_1proc_benchmark(np_arr_shape, mp_queue)

    timings.append(get_timings("baseline", n_frames))
    timings.append(get_timings("queue_module", n_frames))
    timings.append(get_timings("mp_queue", n_frames))

    time_taken = queue_multithread_module(np_arr_shape, mp_queue, n_frames)
    time_summary = {"groupname": "queue_multithread_module",
                    "mean": time_taken / n_frames,
                    "stddev": None,
                    "fps": n_frames / time_taken}
    print(f"{time_summary['groupname']}: time: = {time_summary['mean']} +/- "
          f"{time_summary['stddev']}"
          f" or FPS = {time_summary['fps']}")
    timings.append(time_summary)

    time_taken = mp_queue_multiproc_benchmark(np_arr_shape, mp_queue, n_frames)
    time_summary = {"groupname": "mp_queue_multiproc_benchmark",
                    "mean": time_taken / n_frames,
                    "stddev": None,
                    "fps": n_frames / time_taken}
    print(f"{time_summary['groupname']}: time: = {time_summary['mean']} +/- "
          f"{time_summary['stddev']}"
          f" or FPS = {time_summary['fps']}")
    timings.append(time_summary)

    df = pd.DataFrame(timings)

    return df


if __name__ == "__main__":
    DF = benchmark_queues()
    DF.to_csv("timings/queue_timings.csv")
