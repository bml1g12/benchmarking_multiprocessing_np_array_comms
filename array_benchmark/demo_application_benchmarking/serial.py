"""
The serial implementation as a baseline for processing many streams of frames.
"""
import sys

import cv2
import timing
from tqdm import tqdm

from array_benchmark.shared import prepare_frame

_TIME = timing.get_timing_group(__name__)


def frame_stream(camera_index, array_dim):
    """A demo of a function that is obtaining numpy arrays, and then storing them in a way that
    can be accessed by other processes efficiently. For example, can imagine this represents a
     camera feed  with some processing of the feed.

    :param int camera_index: 0-indexed index specific to each frame stream/camera.
    :param Tuple[int. int] array_dim: dimensions in pixels for the numpy array
    """
    print(f"A generator for processing data from camera id: {camera_index} has been"
          f" instantiated.")
    frames_written = 0
    while True:
        frame = prepare_frame(array_dim, frames_written)
        np_array = frame
        # store img and metadata related to the frame as a tuple
        yield np_array, frames_written
        frames_written += 1


def display_frame_from_camera(frame_gen, show_img):
    """For a given camera"s frame generator, obtain the frame and metadata associated."""
    (np_array, frames_written) = next(frame_gen)  # pylint: disable = unused-variable
    img = np_array.astype("uint8").copy()
    if show_img:
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            sys.exit()
    return img


def benchmark(array_dim, number_of_cameras, show_img, n_frames, repeats):
    """Measure performance of this implementation"""
    print("Serial process started.")
    frame_gens = [frame_stream(selected_camera_index,
                               array_dim) for selected_camera_index in range(number_of_cameras)]

    for timer in _TIME.measure_many("serial", samples=repeats):
        for _ in tqdm(range(n_frames)):
            for camera_index in range(number_of_cameras):
                _ = display_frame_from_camera(frame_gens[camera_index],
                                              show_img)
        timer.stop()

        # for next test
        frame_gens = [frame_stream(selected_camera_index,
                                   array_dim) for selected_camera_index in range(number_of_cameras)]

    # Cleanup
    cv2.destroyAllWindows()
    print("Master process finished.")
    del frame_gens


if __name__ == "__main__":
    benchmark(array_dim=(240, 320), number_of_cameras=2, show_img=True, n_frames=1000, repeats=3)
