"""Benchmark performance of put/getting numpy arrays on a single process"""

import pandas as pd
import timing

from array_benchmark.queue_benchmarking.benchmarks import baseline_benchmark, \
    queue_1thread_module, \
    mp_queue_1proc_benchmark, queue_multithread_module, mp_queue_multiproc_benchmark, \
    mp_queue_multiproc_shared_memory_benchmark
from array_benchmark.shared import get_timings

_TIME = timing.get_timing_group(__name__)


def benchmark_queues():
    """Benchmark various implementations of queues for put/getting numpy arrays"""
    np_arr_shape = (240*4, 320*4)
    n_frames = 1000
    repeats = 3
    metagroupname = "array_benchmark.queue_benchmarking.benchmarks"

    timings = []

    print("Starting baseline timings...")
    baseline_benchmark(np_arr_shape, n_frames, repeats)

    print("Starting queue_1thread_module timings...")
    queue_1thread_module(np_arr_shape, n_frames, repeats)

    print("Starting mp_queue_1proc_benchmark timings...")
    mp_queue_1proc_benchmark(np_arr_shape, n_frames, repeats)

    print("Starting queue_multithread_module timings...")
    queue_multithread_module(np_arr_shape, n_frames, repeats)

    print("Starting mp_queue_multiproc_benchmark timings...")
    mp_queue_multiproc_benchmark(np_arr_shape, n_frames, repeats)

    print("Starting mp_queue_multiproc_shared_memory_benchmark timings...")
    mp_queue_multiproc_shared_memory_benchmark(np_arr_shape, n_frames, repeats)

    timings.append(get_timings(metagroupname, "baseline_benchmark",
                               times_calculated_over_n_frames=n_frames))
    timings.append(get_timings(metagroupname, "queue_1thread_module",
                               times_calculated_over_n_frames=n_frames))
    timings.append(get_timings(metagroupname, "mp_queue_1proc_benchmark",
                               times_calculated_over_n_frames=n_frames))
    timings.append(get_timings(metagroupname, "queue_multithread_module",
                               times_calculated_over_n_frames=n_frames))
    timings.append(get_timings(metagroupname, "mp_queue_multiproc_benchmark",
                               times_calculated_over_n_frames=n_frames))
    timings.append(get_timings(metagroupname, "mp_queue_multiproc_shared_memory_benchmark",
                               times_calculated_over_n_frames=n_frames))

    df = pd.DataFrame(timings)

    return df


if __name__ == "__main__":
    DF = benchmark_queues()
    DF.to_csv("timings/queue_timings.csv")
