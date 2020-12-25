# Benchmarking communication of numpy arrays between Python processes 

This repo compares, in order of speed from slowest to fastest, the following methods for sharing numpy arrays between processes:

1. Serial
2. Simple mp.Queue (serialising and pickling the data into a queue)
3. mp.Array (shared memory) with mp.Queue for metadata
4. mp.Array (shared memory) with mp.Pipe for metadata
