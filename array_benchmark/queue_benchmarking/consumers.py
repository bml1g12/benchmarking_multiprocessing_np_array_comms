"""consumers"""
import time
from tqdm import tqdm

def consumer(n_frames, queue):
    """A frame consumer function, which draws frames from the worker thread/process via a queue
    and does a dummy calculation on the result."""
    for _ in tqdm(range(n_frames)):
        np_arr = queue.get()
        # example of some processing done on the array:
        _ = np_arr.astype("uint8").copy() * 2


def consumer_shared_memory(n_frames, shared_memory):
    """A frame consumer function, which draws frames from the worker process via shared memory."""
    mp_array, np_array = shared_memory
    for _ in tqdm(range(n_frames)):
        _ = np_array.astype("uint8").copy() * 2  # example of some processing done on the array
        while True:
            try:
                mp_array.release()
                break
            # it already unlocked, wait until its locked again which means a new frame is ready
            except ValueError:
                time.sleep(0.001)
