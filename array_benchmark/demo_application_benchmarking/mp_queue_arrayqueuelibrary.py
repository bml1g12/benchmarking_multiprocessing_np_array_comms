"""
The naive and most obvious way to share arrays between processes; a simple queue.
Unfortunately because mp.Queue pickles the numpy array, this is a functional but extremely
slow and expensive way to share numpy arrays between processes.
"""
import multiprocessing as mp
import sys
import time
from queue import Empty, Full

import cv2
import timing
from arrayqueues.shared_arrays import ArrayQueue
from tqdm import tqdm

from array_benchmark.shared import prepare_frame

_TIME = timing.get_timing_group(__name__)


def frame_stream(camera_index, per_camera_array, frame_gen_config):
    """A demo of a function that is obtaining numpy arrays, and then storing them in a way that
    can be accessed by other processes efficiently. For example, can imagine this represents
    a camera feed with some processing of the feed.

    :param int camera_index: 0-indexed index specific to each frame stream/camera.
    :param mp.Queue per_camera_array: Machinery for sharing information between processes,
    but specific to this camera
    :param dict frame_gen_config: A dictionary containing key array_dim, the dimensions
      in pixels for the numpy array as Tuple[int, int]
    """
    print(f"A worker process for processing data from camera id: {camera_index} has started"
          f" processing data in background.")
    (array_queue, metadata_queue) = per_camera_array
    frames_written = 0
    while True:
        frame = prepare_frame(frame_gen_config, frames_written)
        np_array = frame
        # store img and metadata related to the frame as a tuple
        while True:
            try:
                array_queue.put(np_array)
                break
            except Full:
                print("queue full...retrying")
                time.sleep(0.001)
        metadata_queue.put(frames_written)

        frames_written += 1


def setup_mp_resources(frame_gen_config, number_of_cameras):
    """Setup the multiprocessing resources.
     Prepare a queue for each process, used for sharing the frames and the associated metadata
     (together as a tuple) from slave processes to master."""
    procs = []
    per_camera_arrays = {}
    # For each camera, produce create tuples of (multiprocessing.Array, numpy.ndarray)
    # referencing the same underlying buffers
    for camera_index in range(number_of_cameras):
        array_queue = ArrayQueue(100)  # Allocate 300 MB to this array.
        metadata_queue = mp.Queue()
        per_camera_arrays[camera_index] = (array_queue, metadata_queue)
        proc = mp.Process(target=frame_stream,
                          args=(camera_index, per_camera_arrays[camera_index], frame_gen_config))
        procs.append(proc)
    return per_camera_arrays, procs


def display_frame_from_camera(show_img, per_camera_arrays, selected_camera_index):
    """Obtain a frame on master process from worker process with index == selected_camera_index"""
    array_queue, metadata_queue = per_camera_arrays[selected_camera_index]
    while True:
        try:
            np_array = array_queue.get()  # pylint: disable = unused-variable
            break
        except Empty:
            print("queue full...retrying")
            time.sleep(0.001)
    _ = metadata_queue.get()
    img = np_array.astype("uint8").copy()
    if show_img:
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            sys.exit()
    return img


def benchmark(frame_gen_config, number_of_cameras, show_img, n_frames, repeats):
    """Measure performance of this implementation"""
    print("Master process started.")
    per_camera_arrays, procs = setup_mp_resources(frame_gen_config, number_of_cameras)
    for timer in _TIME.measure_many("mp_queue_arrayqueuelibrary", samples=repeats):

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
        print(f"Master process finished: {time2 - time1}")

        # for next test
        per_camera_arrays, procs = setup_mp_resources(frame_gen_config, number_of_cameras)
    del procs
    del per_camera_arrays


if __name__ == "__main__":
    benchmark(frame_gen_config={"array_dim": (240, 320), "is_io_limited": True},
              number_of_cameras=16, show_img=False, n_frames=1000, repeats=3)
