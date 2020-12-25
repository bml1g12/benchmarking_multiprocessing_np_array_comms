"""
A simple approach to using shared memory to transfer numpy array and associated metadata between
processes, inspired by:
https://www.reddit.com/r/learnpython/comments/4u0xh8/fastest_way_to_share_numpy_arrays_between
"""
import multiprocessing as mp
import random
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

from shared import prepare_frame


def frame_stream(camera_index, per_camera_array, array_dim):
    """A demo of a function that is obtaining numpy arrays, and then storing them in a way that
    can be accessed by other processes efficiently. For example, can imagine this represents a
    camera feed with some processing of the feed.

    :param int camera_index: 0-indexed index specific to each frame stream/camera.
    :param tuple per_camera_array: Machinery for sharing information between processes, but specific
    to this camera
    :param Tuple[int. int] array_dim: dimensions in pixels for the numpy array
    """
    print(f"A process for processing data from camera id: {camera_index} has started"
          f" processing data in background.")
    timestamp_pipe, mp_array, np_array = per_camera_array
    frames_written = 0
    while True:
        frame = prepare_frame(array_dim, frames_written)
        mp_array.acquire()
        np_array[:] = frame
        timestamp_pipe["child"].send(frames_written)# store metadata related to the frame, such as timestamp
        frames_written += 1


def setup_mp_resources(array_dim, number_of_cameras):
    """Setup the multiprocessing resources.
    For each camera, produce create tuples of (multiprocessing.Array, numpy.ndarray)
    The numpy array is a view of the multiprocessing Array. Each tuple is specific to each
    worker process (or "camera" if we image each process is processing a camera), and the array
    is being used to share memory between the worker and the master process.

    Here a mp.Pipe is used to communciate associated frame metadata from worker to master.
    """
    procs = []
    per_camera_arrays = {}


    for camera_index in range(number_of_cameras):
        timestamp_pipe = {}
        timestamp_pipe["child"], timestamp_pipe["parent"] = mp.Pipe()
        mp_array = mp.Array("I", int(np.prod(array_dim)), lock=mp.Lock())
        np_array = np.frombuffer(mp_array.get_obj(), dtype="I").reshape(array_dim)
        per_camera_arrays[camera_index] = (timestamp_pipe, mp_array, np_array)
        proc = mp.Process(target=frame_stream,
                          args=(camera_index, per_camera_arrays[camera_index], array_dim))
        procs.append(proc)
    return per_camera_arrays, procs

def display_frame_from_camera(show_img, per_camera_arrays, selected_camera_index):
    """Obtain a frame on master process from worker process with index == selected_camera_index"""
    timestamp_pipe, mp_array, np_array = per_camera_arrays[selected_camera_index]
    frame_metadata = timestamp_pipe["parent"].recv()
    img = np_array.astype("uint8").copy()
    mp_array.release()
    if show_img:
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            sys.exit()
    return img

def benchmark(array_dim, number_of_cameras, show_img):
    """Measure performance of this implementation"""
    print("Master process started.")
    per_camera_arrays, procs = setup_mp_resources(array_dim, number_of_cameras)
    [p.start() for p in procs]

    time1 = time.time()
    for _ in tqdm(range(1000)):
        for camera_index in range(number_of_cameras):
            img = display_frame_from_camera(show_img, per_camera_arrays,
                                       selected_camera_index=camera_index)


    time2 = time.time()
    # Cleanup
    cv2.destroyAllWindows()
    [p.terminate() for p in procs]
    print("Master process finished.")
    return time2-time1

if __name__ == "__main__":
    benchmark(array_dim=(240, 320), number_of_cameras=2, show_img=True)
