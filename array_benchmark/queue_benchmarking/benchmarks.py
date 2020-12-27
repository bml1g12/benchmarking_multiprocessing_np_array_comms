"""Benchmark functions"""
import multiprocessing as mp
import threading
import time

import numpy as np

from array_benchmark.queue_benchmarking.consumers import consumer, consumer_shared_memory
from array_benchmark.queue_benchmarking.producers import worker_producer, \
    worker_producer_shared_memory


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


# def queue_1thread_module_manual_timing(np_arr_shape, queue_module, n_frames):
#     """queue library's queue. All is done on one thread here, and so it seems Python is smart
#     enough not to serialise/pickle the data."""
#     time1 = time.time()
#     for _ in range(n_frames):
#         np_arr = np.random.random(np_arr_shape)  # produce a fresh array
#         queue_module.put(np_arr)
#         np_arr_out = queue_module.get()
#         new_array = np_arr_out * 2  # example of some processing done on the array
#     time2 = time.time()
#     return time2 - time1

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


def mp_queue_multiproc_shared_memory_benchmark(np_arr_shape, n_frames):
    """Passing the numpy array from a producer process to a consumer process via a queue.
    The pickling makes this extremely slow."""
    mp_array = mp.Array("I", int(np.prod(np_arr_shape)), lock=mp.Lock())
    np_array = np.frombuffer(mp_array.get_obj(), dtype="I").reshape(np_arr_shape)
    shared_memory = (mp_array, np_array)
    proc = mp.Process(target=worker_producer_shared_memory,
                      args=(np_arr_shape, shared_memory, n_frames))
    proc.start()
    time1 = time.time()
    consumer_shared_memory(n_frames, shared_memory)  # will consume n_frames from producer
    time2 = time.time()
    proc.terminate()
    return time2 - time1
