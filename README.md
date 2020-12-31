# Benchmarking communication of numpy arrays between Python processes 

Tested using Python 3.7.0 on Ubuntu 20.04 LTS with Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz (4-core, 8-thread CPU)   

This repo compares, in order of speed from slowest to fastest, the following methods for sharing numpy arrays between processes:

1. Serial baseline
2. Simple mp.Queue (serialising and pickling the data into a queue)
3. mp.Array to avoid pickling in a custom Queue implementation from: https://github.com/portugueslab/arrayqueues
4. mp.Array (shared memory) with mp.Queue for metadata
5. mp.Array (shared memory) with mp.Pipe for metadata
6. threading.Thread with queue.Queue for sharing arrays.

It implements two benchmarks:

1. ("queue_benchmarking") Passing an image numpy array from a producer to a consumer
    * A (240, 320) array over 1000 frames
2. ("demo_application_benchmarking") Passing an image numpy array from many producers to a single consumer, which could then go on to, for example, stich the images into a real-time grid.
    * A (240, 320) array from 16 producers to a signle consumer over 1000 frames.
    
And includes a few implementations of the same the [Ray library](https://docs.ray.io/en/latest/index.html).

In these benchmarks, an `time.sleep()` is used on the Producer; when this occurs during the multithreading benchmarks, other threads can jump into action and make use of the CPU, so its a good emulation of I/O bound Producer. As a result, multithreading comes out very fast here, if we have a CPU bound Producer then we can likely expect multiprocessing to trump the max speed.

## How To Run 

`pip install requirements.txt`

`pip install -e setup.py`

`python array_benchmark/demo_application_benchmarking/main.py`

`python array_benchmark/queue_benchmarking/main.py`

## Output

Timings can be found in the ./timings folder.

Timings are reported over 1000 frames as `time_for_all_frames` (seconds) +/- `stddev_for_all_frames` (seconds)  with this standard deviation calculatied over 3 repeats. `time_per_frame` is calculated as `time_for_all_frames`/1000 and the FPS is calculated as 1/`time_per_frame`.


