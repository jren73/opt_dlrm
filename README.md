# optimizing dlrm inference

optimizing dlrm inference by caching/prefetching indices from host cpu main memory/ssd to GPU mem.

## Files
 - Preprocessing dataset: optgen.py, lru.py. input with indices trace, output the cached traces with opt, LRU and LFU caching traces and cache miss traces. Those traces will be used as model training ground-truth

 - training caching model: seq2seq_caching.py. Configurable parameters:
   - input sequence length N and output sequence length M
   - ground-truth: opt cache trace

 - training prefetching model: seq2seq_prefetching.py. Configurable parameters:
   - input sequence length N, output sequence length M, evaluation window size W
   - loss function: mean_squared_error or IoU loss
   - number of lstm stacks
