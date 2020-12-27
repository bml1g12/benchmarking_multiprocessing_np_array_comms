"""producers"""
import numpy as np


def worker_producer(np_arr_shape, queue, n_frames):
    """A frame producer function, e.g. for a worker thread or process"""
    for _ in range(n_frames):
        np_arr = np.random.random(np_arr_shape)  # produce a fresh array
        queue.put(np_arr)


def worker_producer_shared_memory(np_arr_shape, shared_memory, n_frames):
    """A frame producer function that writes to shared memory"""
    mp_array, np_array = shared_memory
    for _ in range(n_frames):
        mp_array.acquire()
        np_array[:] = np.random.random(np_arr_shape)  # produce a fresh array
