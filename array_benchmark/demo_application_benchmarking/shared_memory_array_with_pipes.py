"""
A simple approach to using shared memory to transfer numpy array and associated metadata between
processes

Here for passing metadata related to the array, we use a multiprocessing pipe which is more
efficient than a mulitprocessing Queue for communication between a pair of processes.
"""
import multiprocessing as mp
import sys
import time

import cv2
import numpy as np
import timing
from tqdm import tqdm

from array_benchmark.shared import prepare_frame

_TIME = timing.get_timing_group(__name__)


def frame_stream(camera_index, per_camera_array, array_dim):
    """A demo of a function that is obtaining numpy arrays, and then storing them in a way that
    can be accessed by other processes efficiently. For example, can imagine this represents a
    camera feed with some processing of the feed.

    :param int camera_index: 0-indexed index specific to each frame stream/camera.
    :param tuple per_camera_array: Machinery for sharing information between processes, but specific
    to this camera
    :param Tuple[int. int] array_dim: dimensions in pixels for the numpy array
    """
    print(f"A worker process for processing data from camera id: {camera_index} has started"
          f" processing data in background.")
    timestamp_pipe, mp_array, np_array = per_camera_array
    frames_written = 0
    while True:
        frame = prepare_frame(array_dim, frames_written)
        mp_array.acquire()
        np_array[:] = frame
        # store metadata related to the frame, such as timestamp
        timestamp_pipe["child"].send(frames_written)
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
    # get the frame metadata
    _ = timestamp_pipe["parent"].recv()
    img = np_array.astype("uint8").copy()
    if show_img:
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            sys.exit()
    mp_array.release()
    return img


def benchmark(array_dim, number_of_cameras, show_img, n_frames, repeats):
    """Measure performance of this implementation"""
    print("Master process started.")
    per_camera_arrays, procs = setup_mp_resources(array_dim, number_of_cameras)
    for timer in _TIME.measure_many("shared_memory_array_with_pipes", samples=repeats):
        for proc in procs:
            proc.start()

        time1 = time.time()
        for _ in tqdm(range(n_frames)):
            for camera_index in range(number_of_cameras):
                _ = display_frame_from_camera(show_img, per_camera_arrays,
                                              selected_camera_index=camera_index)
        timer.stop()
        time2 = time.time()
        # Cleanup
        cv2.destroyAllWindows()
        for proc in procs:
            proc.terminate()
        print(f"Master process finished: {time2-time1}")
        # for next test
        per_camera_arrays, procs = setup_mp_resources(array_dim, number_of_cameras)
    del per_camera_arrays
    del procs


if __name__ == "__main__":
    benchmark(array_dim=(240, 320), number_of_cameras=2, show_img=False, n_frames=1000, repeats=3)
