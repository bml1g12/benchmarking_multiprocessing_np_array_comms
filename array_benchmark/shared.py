"""Shared functions between benchmarks"""

import time

import numpy as np
import timing


def prepare_frame(frame_gen_config, frames_written):
    """Emulating a time consuming process of obtaining a frame
    frame_gen_config is a dictionary with two keys: array_dim and is_io_limited.
    :param Tuple[int, int] array_dim: dimensions of the frame we wish to produce
    :param int frames_written: number of frames written so far, used to gradually increase the
    whiteness of the frame, so we can "see" chronology in a video of the output.
    :param bool is_io_limited: If True, will use time.sleep() to emulate an I/O limited producer.
    If False will use a CPU heavy calculation to emulate a CPU limited producer.
    :returns np.array frame: A numpy array of the producer frame
    """
    frame = np.ones(frame_gen_config["array_dim"]) * frames_written
    if frame_gen_config["is_io_limited"]:
        time.sleep(0.001)
    else:
        time1 = time.time()
        while True:
            _ = 999*999
            time2 = time.time()
            if (time2-time1) > 0.001:
                break

    return frame


def get_timings(metagroupname, groupname, times_calculated_over_n_frames):
    """ Get a dictionary of the mean/std and FPS of the timing group.

    :param str metagroupname: The module path for this timing group
    :param str groupname: name of the timing group function name
    :param int times_calculated_over_n_frames: The _TIME["mean"] corresponds to this integer worth
    of frames
    :return: mean/std and FPS of the timing group as a dictionary
    :rtype: dict
    """
    # mean is the time per frame in this code
    timing_group = timing.get_timing_group(metagroupname)
    time_per_frame = timing_group.summary[groupname]["mean"]/times_calculated_over_n_frames
    stddev = f"{timing_group.summary[groupname]['stddev']:.4f}"
    fps = f"{1 / time_per_frame}"
    print(f"{groupname}: time_for_all_frames: = {timing_group.summary[groupname]['mean']} +/- "
          f"{stddev}"
          f" or FPS = {fps}")
    return {"groupname": groupname,
            "time_per_frame": f"{time_per_frame:.4f}",
            "time_for_all_frames": timing_group.summary[groupname]["mean"],
            "stddev_for_all_frames": stddev,
            "fps": fps}
