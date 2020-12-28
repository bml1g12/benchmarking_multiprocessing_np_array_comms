"""Benchmark functions"""
import inspect
import multiprocessing as mp
import threading


from queue import Queue
from tqdm import tqdm


import numpy as np
import timing

from array_benchmark.queue_benchmarking.consumers import consumer, consumer_shared_memory
from array_benchmark.queue_benchmarking.producers import worker_producer, \
    worker_producer_shared_memory, prepare_random_frame


_TIME = timing.get_timing_group(__name__)


def baseline_benchmark(np_arr_shape, n_frames, repeats):
    """baseline"""
    for _ in _TIME.measure_many(inspect.currentframe().f_code.co_name, samples=repeats):
        for _ in tqdm(range(n_frames)):
            np_arr = prepare_random_frame(np_arr_shape)  # produce a fresh array
            # example of some processing done on the array
            _ = np_arr.astype("uint8").copy() * 2


def queue_1thread_module(np_arr_shape, n_frames, repeats):
    """queue library's queue. All is done on one thread here, and so it seems Python is smart
    enough not to serialise/pickle the data."""
    queue_module = Queue()
    for timer in _TIME.measure_many(inspect.currentframe().f_code.co_name, samples=repeats):
        for _ in tqdm(range(n_frames)):
            np_arr = np.random.random(np_arr_shape)  # produce a fresh array
            queue_module.put(np_arr)
            np_arr_out = queue_module.get()
            # example of some processing done on the array
            _ = np_arr_out.astype("uint8").copy() * 2
        timer.stop()
        # for next test
        queue_module = Queue()
    del queue_module

def queue_multithread_module(np_arr_shape, n_frames, repeats):
    """Passing the numpy array from a producer thread to a consumer thread via a Queue.queue.
    The queue.Queue library likely is not  pickling as its fast."""
    queue = Queue()

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name, samples=repeats)):
        thread = threading.Thread(target=worker_producer,
                                  args=(np_arr_shape, queue, n_frames))
        thread.start()
        consumer(n_frames, queue)  # will consume n_frames from producer
        timer.stop()
        thread.join()
        # for next test
        queue = Queue()
    del queue


def mp_queue_1proc_benchmark(np_arr_shape, n_frames, repeats):
    """multiprocessing queues. All is done on one process here, and so it seems Python is smart
    enough not to serialise/pickle the data.

    I believe this is particularly slow as it takes a while for an item to be put onto the queue
    due to the pickling, and this creates a constant bottleneck when the benchmark here involves
    a blocking get after every put. In contrast to a constant queue of puts and gets.
    """
    mp_queue = mp.Queue()
    for timer in _TIME.measure_many(inspect.currentframe().f_code.co_name, samples=repeats):
        for _ in tqdm(range(n_frames)):
            np_arr = prepare_random_frame(np_arr_shape)  # produce a fresh array
            mp_queue.put(np_arr)
            np_arr = mp_queue.get()
            # example of some processing done on the array
            _ = np_arr.astype("uint8").copy() * 2
        timer.stop()
        # for next test
        mp_queue = mp.Queue()
    del mp_queue

def mp_queue_multiproc_benchmark(np_arr_shape, n_frames, repeats):
    """Passing the numpy array from a producer process to a consumer process via a queue.
    The pickling makes this extremely slow."""
    mp_queue = mp.Queue()

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name, samples=repeats)):
        proc = mp.Process(target=worker_producer,
                          args=(np_arr_shape, mp_queue, n_frames))
        proc.start()
        consumer(n_frames, mp_queue)  # will consume n_frames from producer
        timer.stop()
        proc.terminate()
        # for next test
        mp_queue = mp.Queue()
    del mp_queue

def mp_queue_multiproc_shared_memory_benchmark(np_arr_shape, n_frames, repeats):
    """Passing the numpy array from a producer process to a consumer process via a queue.
    The pickling makes this extremely slow."""
    mp_array = mp.Array("I", int(np.prod(np_arr_shape)), lock=mp.Lock())
    np_array = np.frombuffer(mp_array.get_obj(), dtype="I").reshape(np_arr_shape)
    shared_memory = (mp_array, np_array)

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name, samples=repeats)):
        proc = mp.Process(target=worker_producer_shared_memory,
                          args=(np_arr_shape, shared_memory, n_frames))
        proc.start()
        consumer_shared_memory(n_frames, shared_memory)  # will consume n_frames from producer
        timer.stop()
        proc.terminate()
        # for next test
        mp_array = mp.Array("I", int(np.prod(np_arr_shape)), lock=mp.Lock())
        np_array = np.frombuffer(mp_array.get_obj(), dtype="I").reshape(np_arr_shape)
        shared_memory = (mp_array, np_array)

    del mp_array, np_array, shared_memory
