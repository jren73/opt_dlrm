# optimizing dlrm inference

optimizing dlrm inference by caching/prefetching indices from host cpu main memory/ssd to GPU mem.

## Files
 - Preprocessing dataset: optgen.py, lru.py. 
  - optgen.py: generating cache hit and miss trace with opt caching (simulating belady's algorithm) strategy. Time complaxity: O(N^2) 
    - input: indices trace; 
    - output: (1) tracename_cached_trace_opt.csv, includes an binary array, where 1 stands for cache hit with opt caching strategy. (2) tracename__dataset_cache_miss_trace.csv, includes indices which have cache miss. Zero stands for the corresponding index has cache hit
    - configurable parameter: (1) cache size in percent: we 


  - lru.py: generating cache hit and miss trace with LRU and LFU caching strategy. Time complaxity: O(N) 
     - input: indices trace; 
     - output: (1) tracename_lru_cache.csv, includes an binary array, where 1 stands for cache hit with LRU caching strategy. (2) tracename_lru_miss.csv, includes indices which have cache miss. Zero stands for the corresponding index has cache hit

 input with indices trace, output the cached traces with opt, LRU and LFU caching traces and cache miss traces. Those traces will be used as model training ground-truth

 - training caching model: seq2seq_caching.py. Configurable parameters:
   - input sequence length N and output sequence length M
   - ground-truth: opt cache trace

 - training prefetching model: seq2seq_prefetching.py. Configurable parameters:
   - input sequence length N, output sequence length M, evaluation window size W
   - loss function: mean_squared_error or IoU loss
   - number of lstm stacks
