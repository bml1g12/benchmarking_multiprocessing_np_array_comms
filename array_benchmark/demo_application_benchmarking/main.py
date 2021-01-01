"""Benchmarks different implementations for sharing numpy arrays between processes."""
# pylint: disable=expression-not-assigned
import pandas as pd

import array_benchmark.demo_application_benchmarking.multithreaded_queue as multithreaded_queue
import array_benchmark.demo_application_benchmarking.naive_mp_queue as naive_mp_queue
import array_benchmark.demo_application_benchmarking.serial as serial
import array_benchmark.demo_application_benchmarking.shared_memory_array as shared_memory_array
import array_benchmark.demo_application_benchmarking.shared_memory_array_with_pipes as \
    shared_memory_array_with_pipes
#import array_benchmark.demo_application_benchmarking.mp_queue_arrayqueuelibrary as \
#    mp_queue_arrayqueuelibrary
from array_benchmark.shared import get_timings


def main():
    """Main benchmarking script"""
    array_dim = (240, 320)
    n_frames = 1000
    number_of_cameras = 16
    repeats = 3
    show_img = False
    metagroupname = "array_benchmark.demo_application_benchmarking"
    frame_gen_config = {
        "array_dim": array_dim,
        # if True, producer emulates I/O bound (sleep) if False, emulate CPU bound
        "is_io_limited": True
    }

    serial.benchmark(frame_gen_config, number_of_cameras, show_img, n_frames, repeats)
    multithreaded_queue.benchmark(frame_gen_config, number_of_cameras, show_img, n_frames, repeats)
    naive_mp_queue.benchmark(frame_gen_config, number_of_cameras, show_img, n_frames, repeats)
    #mp_queue_arrayqueuelibrary.benchmark(frame_gen_config, number_of_cameras,
    #                                     show_img, n_frames, repeats)
    shared_memory_array.benchmark(frame_gen_config, number_of_cameras, show_img, n_frames, repeats)
    shared_memory_array_with_pipes.benchmark(frame_gen_config, number_of_cameras,
                                             show_img, n_frames, repeats)

    timings = [get_timings(metagroupname + ".serial", "serial",
                           times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".multithreaded_queue", "multithreaded_queue",
                           times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".naive_mp_queue", "naive_mp_queue",
                           times_calculated_over_n_frames=n_frames),
               #get_timings(metagroupname + ".mp_queue_arrayqueuelibrary",
               #            "mp_queue_arrayqueuelibrary",
               #            times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".shared_memory_array", "shared_memory_array",
                           times_calculated_over_n_frames=n_frames),
               get_timings(metagroupname + ".shared_memory_array_with_pipes",
                           "shared_memory_array_with_pipes",
                           times_calculated_over_n_frames=n_frames)]

    df = pd.DataFrame(timings)
    if frame_gen_config["is_io_limited"]:
        filename = "timings/benchmark_timings_iolimited.csv"
    else:
        filename = "timings/benchmark_timings_cpulimited.csv"
    df.to_csv(filename)


if __name__ == "__main__":
    main()
