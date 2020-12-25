"""Shared code between benchmarks"""
import time

import numpy as np


def prepare_frame(array_dim, frames_written):
    """Emulating a time consuming process of obtaining a frame"""
    frame = np.ones(array_dim) * frames_written
    time.sleep(0.001)
    return frame
