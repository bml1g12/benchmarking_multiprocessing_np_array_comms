# Benchmarking communication of numpy arrays between Python processes 

Tested using Python 3.7.0 on Ubuntu 20.04 LTS with Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz (4-core, 8-thread CPU)   

This repo compares, in order of speed from slowest to fastest, the following methods for sharing numpy arrays between processes:

1. Serial baseline
2. Simple mp.Queue (serialising and pickling the data into a queue)
3. mp.Array to avoid pickling in a custom Queue implementation from: https://github.com/portugueslab/arrayqueues
4. mp.Array (shared memory) with mp.Queue for metadata
5. mp.Array (shared memory) with mp.Pipe for metadata
6. threading.Thread with queue.Queue for sharing arrays.
