"""Benchmarks different implementations for sharing numpy arrays between processes."""
# pylint: disable=expression-not-assigned
import pandas as pd

import array_benchmark.demo_application_benchmarking.multithreaded_queue as multithreaded_queue
import array_benchmark.demo_application_benchmarking.naive_mp_queue as naive_mp_queue
import array_benchmark.demo_application_benchmarking.serial as serial
import array_benchmark.demo_application_benchmarking.shared_memory_array as shared_memory_array
import \
    array_benchmark.demo_application_benchmarking.shared_memory_array_with_pipes as \
        shared_memory_array_with_pipes
from array_benchmark.shared import get_timings


def main():
    """Main benchmarking script"""
    array_dim = (240, 320)
    n_frames = 1000
    number_of_cameras = 16
    repeats = 3
    show_img = False
    metagroupname = "array_benchmark.demo_application_benchmarking"

    serial.benchmark(array_dim, number_of_cameras, show_img, n_frames, repeats)
    multithreaded_queue.benchmark(array_dim, number_of_cameras, show_img, n_frames, repeats)
    naive_mp_queue.benchmark(array_dim, number_of_cameras, show_img, n_frames, repeats)
    shared_memory_array.benchmark(array_dim, number_of_cameras, show_img, n_frames, repeats)
    shared_memory_array_with_pipes.benchmark(array_dim, number_of_cameras,
                                             show_img, n_frames, repeats)

    timings = [get_timings(metagroupname + ".serial", "serial",
                           times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".multithreaded_queue", "multithreaded_queue",
                           times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".naive_mp_queue", "naive_mp_queue",
                           times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".shared_memory_array", "shared_memory_array",
                           times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".shared_memory_array_with_pipes",
                           "shared_memory_array_with_pipes",
                           times_calculated_over_n_frames=n_frames)]

    df = pd.DataFrame(timings)
    df.to_csv("timings/benchmark_timings.csv")


if __name__ == "__main__":
    main()
