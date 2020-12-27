"""Benchmark performance of put/getting numpy arrays on a single process"""

import multiprocessing as mp
from queue import Queue

import pandas as pd
import timing

from array_benchmark.queue_benchmarking.benchmarks import baseline_benchmark, \
    queue_1thread_module, \
    mp_queue_1proc_benchmark, queue_multithread_module, mp_queue_multiproc_benchmark, \
    mp_queue_multiproc_shared_memory_benchmark


_TIME = timing.get_timing_group(__name__)


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
    print(f"FPS is calculated as {n_frames} / {_TIME.summary[groupname]['mean']}" )
    print(f"{groupname}: time: = {mean} +/- "
          f"{stddev}"
          f" or FPS = {fps}")
    return {"groupname": groupname, "mean": mean, "stddev": stddev, "fps": fps}


def benchmark_queues():
    """Benchmark various implementations of queues for put/getting numpy arrays"""
    np_arr_shape = (240, 320)
    n_frames = 1000
    mp_queue = mp.Queue()
    queue_module = Queue()
    timings = []

    for _ in _TIME.measure_many("baseline", samples=n_frames, threshold=3):
        baseline_benchmark(np_arr_shape)

    for _ in _TIME.measure_many("queue_1thread_module", samples=n_frames, threshold=3):
        queue_1thread_module(np_arr_shape, queue_module)

    for _ in _TIME.measure_many("mp_queue_1proc_benchmark", samples=n_frames, threshold=3):
        mp_queue_1proc_benchmark(np_arr_shape, mp_queue)

    timings.append(get_timings("baseline", n_frames))
    timings.append(get_timings("queue_1thread_module", n_frames))
    timings.append(get_timings("mp_queue_1proc_benchmark", n_frames))

    time_taken = queue_multithread_module(np_arr_shape, mp_queue, n_frames)
    time_summary = {"groupname": "queue_multithread_module",
                    "mean": time_taken / n_frames,
                    "stddev": None,
                    "fps": n_frames / (time_taken / n_frames)}
    print(f"{time_summary['groupname']}: time: = {time_summary['mean']} +/- "
          f"{time_summary['stddev']}"
          f" or FPS = {time_summary['fps']}")
    timings.append(time_summary)

    time_taken = mp_queue_multiproc_benchmark(np_arr_shape, mp_queue, n_frames)
    time_summary = {"groupname": "mp_queue_multiproc_benchmark",
                    "mean": time_taken / n_frames,
                    "stddev": None,
                    "fps": n_frames / (time_taken / n_frames)}
    print(f"{time_summary['groupname']}: time: = {time_summary['mean']} +/- "
          f"{time_summary['stddev']}"
          f" or FPS = {time_summary['fps']}")
    timings.append(time_summary)

    time_taken = mp_queue_multiproc_shared_memory_benchmark(np_arr_shape, n_frames)
    time_summary = {"groupname": "mp_queue_multiproc_shared_memory_benchmark",
                    "mean": time_taken / n_frames,
                    "stddev": None,
                    "fps": n_frames / (time_taken / n_frames)}
    print(f"{time_summary['groupname']}: time: = {time_summary['mean']} +/- "
          f"{time_summary['stddev']}"
          f" or FPS = {time_summary['fps']}")
    timings.append(time_summary)

    df = pd.DataFrame(timings)

    return df


if __name__ == "__main__":
    DF = benchmark_queues()
    DF.to_csv("timings/queue_timings.csv")
