"""producers"""
import time

import numpy as np


def prepare_random_frame(np_arr_shape):
    """Emulate an expensive frame generation/processing step here, as if the computation
    for a given process is trivial, then we might as well not use processes and do it in serial.

    :param np_arr_shape Tuple[int, int]: array shape
    :returns: a random generated black and white np.array
    """
    np_arr = np.random.random(np_arr_shape)  # produce a fresh array
    time.sleep(0.001)
    return np_arr


def worker_producer(np_arr_shape, queue, n_frames):
    """A frame producer function, e.g. for a worker thread or process
    :param Tuple[int, int] np_arr_shape: The shape of the numpy array to generate
    :param int n_frames: the number of frames to produce
    """
    for _ in range(n_frames):
        # produce a fresh array
        np_arr = prepare_random_frame(np_arr_shape)
        queue.put(np_arr)


def worker_producer_shared_memory(np_arr_shape, shared_memory, n_frames):
    """A frame producer function that writes to shared memory"""
    mp_array, np_array = shared_memory
    for _ in range(n_frames):
        mp_array.acquire()
        np_array[:] = prepare_random_frame(np_arr_shape)  # produce a fresh array


def worker_producer_arrayqueuelibrary(np_arr_shape, queue, n_frames):
    """A frame producer function, e.g. for a worker thread or process"""
    for _ in range(n_frames):
        np_arr = prepare_random_frame(np_arr_shape)  # produce a fresh array
        queue.put(np_arr)
