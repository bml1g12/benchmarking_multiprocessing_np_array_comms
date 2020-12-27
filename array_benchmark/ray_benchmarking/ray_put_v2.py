"""
Trying to avoid this issue
https://docs.ray.io/en/master/auto_examples/tips-for-first-time.html#tip-4-pipeline-data-processing
by using ray.wait

This results in a fast implemnatation with something like a backlog of items being prepared and
and master worker pulling off the backlog simultaneously, but would need to sort the output as
its being processed in order of completion not chronologically. Sorting the output without
hitting a memory issue might not be trivial, because in current implementation shared memory is
cleared ASAP after its obtained.

It is still slightly slower than shared_memory_array_with_pipes.py.
"""
import ray
from ray.util.queue import Queue as RayQueue
import sys
import time

import cv2

from array_benchmark.shared import prepare_frame

ray.init()

@ray.remote
class FrameStreamWorker(object):
    def __init__(self, camera_index, per_camera_array, array_dim):
        """A demo of a function that is obtaining numpy arrays, and then storing them in a way that
        can be accessed by other processes efficiently. For example, can imagine this represents
        a camera feed with some processing of the feed.

        :param int camera_index: 0-indexed index specific to each frame stream/camera.
        :param tuple per_camera_array: Machinery for sharing information between processes,
        but specific
        to this camera
        :param Tuple[int. int] array_dim: dimensions in pixels for the numpy array
        """
        self.queue = per_camera_array
        self.frames_written = 0
        self.array_dim = array_dim
        self.camera_index = camera_index
        print(f"A worker process for processing data from camera id: {camera_index} has started"
              f" processing data in background.")

    def get_frame(self):
        self.frame = prepare_frame(self.array_dim, self.frames_written)
        np_array = self.frame
        self.frames_written += 1
        return np_array # self.frames_written


def setup_mp_resources(array_dim, number_of_cameras):
    """Setup the multiprocessing resources.
     Prepare a queue for each process, used for sharing the frames and the associated metadata
     (together as a tuple) from slave processes to master."""
    procs = []
    per_camera_arrays = {}
    # For each camera, produce create tuples of (multiprocessing.Array, numpy.ndarray)
    # referencing the same underlying buffers
    for camera_index in range(number_of_cameras):
        queue = RayQueue(maxsize=100)
        per_camera_arrays[camera_index] = queue
        proc = FrameStreamWorker.remote(camera_index, per_camera_arrays[camera_index], array_dim)
        procs.append(proc)
    return per_camera_arrays, procs


def display_frame_from_camera(selected_proc, show_img, per_camera_arrays, selected_camera_index):
    """Obtain a frame on master process from worker process with index == selected_camera_index"""
    obj_ref = selected_proc.get_frame.remote()
    np_array = ray.get(obj_ref)
    img = np_array.astype("uint8").copy()
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

    time1 = time.time()
    # launch a batch of 100 jobs (frame gens(, process the results,
    # then go onto the next batch of 100

    refs2camid = {procs[camera_index].get_frame.remote(): camera_index for _ in range(1000) for
                  camera_index, _ in enumerate(procs)}

    result_ids = list(refs2camid.keys())

    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        obj_ref = done_id[0]
        camera_index = refs2camid[obj_ref]
        np_array = ray.get(obj_ref)
        if show_img:
            cv2.imshow("img", np_array.astype("uint8").copy())
            k = cv2.waitKey(1)
            if k == ord("q"):
                sys.exit()
        del obj_ref
        del np_array

    time2 = time.time()
    # Cleanup
    cv2.destroyAllWindows()

    print(f"Master process finished: {time2-time1}")
    return time2-time1


if __name__ == "__main__":
    benchmark(array_dim=(240, 320), number_of_cameras=2, show_img=False)
