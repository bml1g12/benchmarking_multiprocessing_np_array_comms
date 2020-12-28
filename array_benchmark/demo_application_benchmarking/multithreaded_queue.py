"""
The naive and most obvious way to share arrays between processes; a simple queue.
Unfortunately because mp.Queue pickles the numpy array, this is a functional but extremely
slow and expensive way to share numpy arrays between processes.
"""
import sys
import threading
import time
from queue import Queue

import cv2
import timing
from tqdm import tqdm

from array_benchmark.shared import prepare_frame

_TIME = timing.get_timing_group(__name__)


def frame_stream(camera_index, per_camera_array, array_dim, n_frames):
    """A demo of a function that is obtaining numpy arrays, and then storing them in a way that
    can be accessed by other processes efficiently. For example, can imagine this represents
    a camera feed with some processing of the feed.

    :param int camera_index: 0-indexed index specific to each frame stream/camera.
    :param queue.Queue per_camera_array: Machinery for sharing information between processes,
    but specific to this camera
    :param Tuple[int. int] array_dim: dimensions in pixels for the numpy array
    :param int n_frames: how many frames to write before killing self
    """
    print(f"A worker process for processing data from camera id: {camera_index} has started"
          f" processing data in background.")
    queue = per_camera_array
    frames_written = 0
    while True:
        frame = prepare_frame(array_dim, frames_written)
        np_array = frame
        # store img and metadata related to the frame as a tuple
        queue.put((np_array, frames_written))
        frames_written += 1
        if frames_written == n_frames:
            break


def setup_mp_resources(array_dim, number_of_cameras, n_frames):
    """Setup the multiprocessing resources.
     Prepare a queue for each process, used for sharing the frames and the associated metadata
     (together as a tuple) from slave processes to master."""
    threads = []
    per_camera_arrays = {}
    # For each camera, produce create tuples of (multiprocessing.Array, numpy.ndarray)
    # referencing the same underlying buffers
    for camera_index in range(number_of_cameras):
        queue = Queue(maxsize=100)
        per_camera_arrays[camera_index] = queue
        thread = threading.Thread(target=frame_stream,
                                  args=(camera_index,
                                        per_camera_arrays[camera_index],
                                        array_dim,
                                        n_frames))
        threads.append(thread)
    return per_camera_arrays, threads


def display_frame_from_camera(show_img, per_camera_arrays, selected_camera_index):
    """Obtain a frame on master process from worker process with index == selected_camera_index"""
    queue = per_camera_arrays[selected_camera_index]
    (np_array, frame_metadata) = queue.get()  # pylint: disable = unused-variable
    img = np_array.astype("uint8").copy()
    if show_img:
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            sys.exit()
    return img


def benchmark(array_dim, number_of_cameras, show_img, n_frames, repeats):
    """Measure performance of this implementation"""
    print("Master thread started.")
    per_camera_arrays, threads = setup_mp_resources(array_dim, number_of_cameras, n_frames)
    for timer in _TIME.measure_many("multithreaded_queue", samples=repeats):
        for thread in threads:
            thread.start()

        time1 = time.time()
        for _ in tqdm(range(n_frames)):
            for camera_index in range(number_of_cameras):
                _ = display_frame_from_camera(show_img,
                                              per_camera_arrays,
                                              selected_camera_index=camera_index)

        timer.stop()
        time2 = time.time()
        print(f"Master thread finished: {time2 - time1}")
        # Cleanup
        cv2.destroyAllWindows()
        for thread in threads:
            thread.join()

        # for next test
        per_camera_arrays, threads = setup_mp_resources(array_dim, number_of_cameras, n_frames)
    del per_camera_arrays
    del threads


if __name__ == "__main__":
    benchmark(array_dim=(240, 320), number_of_cameras=2, show_img=False, n_frames=1000, repeats=3)
