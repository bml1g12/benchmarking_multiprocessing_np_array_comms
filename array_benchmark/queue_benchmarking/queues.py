import multiprocessing as mp
from queue import Queue

import numpy as np
import timing

_TIME = timing.get_timing_group(__name__)

def baseline_benchmark(np_arr):
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array

def mp_queue_benchmark(np_arr, mp_queue):
    mp_queue.put(np_arr)
    np_arr = mp_queue.get()
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array

def queue_module_benchmark(np_arr, queue_module):
    queue_module.put(np_arr)
    np_arr = queue_module.get()
    new_array = np_arr * 2  # example of some processing done on the array
    return new_array

def benchmark_queues():
    np_arr = np.ones((1000, 1000))
    n_frames = 100
    mp_queue = mp.Queue()
    queue_module = Queue()

    for timer in _TIME.measure_many('baseline', samples=n_frames):
        baseline_benchmark(np_arr)

    for timer in _TIME.measure_many('mp_queue', samples=n_frames):
        mp_queue_benchmark(np_arr, mp_queue)

    for timer in _TIME.measure_many('queue_module', samples=n_frames):
        queue_module_benchmark(np_arr, queue_module)

    print(f"baseline: time: = {_TIME.summary['baseline']['mean']:.4f} +/- "
          f"{_TIME.summary['baseline']['stddev']:.4f}"
          f" or FPS = {n_frames/_TIME.summary['baseline']['mean']}")

    print(f"mp.Queue: time: = {_TIME.summary['mp_queue']['mean']:.4f} +/- "
          f"{_TIME.summary['mp_queue']['stddev']:.4f}"
          f" or FPS = {n_frames/_TIME.summary['mp_queue']['mean']}")

    print(f"queue_module.Queue: time: = {_TIME.summary['queue_module']['mean']:.4f} +/- "
          f"{_TIME.summary['queue_module']['stddev']:.4f} "
          f"or FPS = {n_frames/_TIME.summary['queue_module']['mean']}")

if __name__ == "__main__":
    benchmark_queues()

