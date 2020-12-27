"""Benchmarks different implementations for sharing numpy arrays between processes."""
# pylint: disable=expression-not-assigned
import pandas as pd

import array_benchmark.demo_application_benchmarking.naive_mp_queue as naive_mp_queue
import array_benchmark.demo_application_benchmarking.serial as serial
import array_benchmark.demo_application_benchmarking.shared_memory_array as shared_memory_array
import \
    array_benchmark.demo_application_benchmarking.shared_memory_array_with_pipes as \
        shared_memory_array_with_pipes


def main():
    """Main benchmarking script"""
    array_dim = (240, 320)
    number_of_cameras = 16
    show_img = False

    timings = []
    for _ in range(10):
        row = {}
        time_taken = serial.benchmark(array_dim, number_of_cameras, show_img)
        print(f"serial: {time_taken}")
        row["serial"] = time_taken

        time_taken = naive_mp_queue.benchmark(array_dim, number_of_cameras, show_img)
        print(f"naive_mp_queue: {time_taken}")
        row["naive_mp_queue"] = time_taken

        time_taken = shared_memory_array.benchmark(array_dim, number_of_cameras, show_img)
        print(f"shared_memory_array: {time_taken}")
        row["shared_memory_array"] = time_taken

        time_taken = shared_memory_array_with_pipes.benchmark(array_dim, number_of_cameras,
                                                              show_img)
        print(f"shared_memory_array_with_pipes: {time_taken}")
        row["shared_memory_array_with_pipes"] = time_taken
        timings.append(row)
    df = pd.DataFrame(timings)
    df.to_csv("timings/benchmark_timings.csv")


if __name__ == "__main__":
    main()
