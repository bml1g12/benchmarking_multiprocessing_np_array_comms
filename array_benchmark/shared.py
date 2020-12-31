"""Shared functions between benchmarks"""

import time

import numpy as np
import timing


def prepare_frame(array_dim, frames_written):
    """Emulating a time consuming process of obtaining a frame"""
    frame = np.ones(array_dim) * frames_written
    time.sleep(0.001)
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
