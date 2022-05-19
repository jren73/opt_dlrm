# optimizing dlrm inference

optimizing dlrm inference by caching/prefetching indices from host cpu main memory/ssd to GPU mem.

## Files
 - Preprocessing dataset: optgen.py, lru.py. Generating models' training ground-truth
    - optgen.py: generating cache hit and miss trace with opt caching (simulating belady's algorithm) strategy. Time complaxity: O(N^2) 
      - input: indices trace; 
      - output: (1) tracename_cached_trace_opt.csv, includes an binary array, where 1 stands for cache hit with opt caching strategy. (2) tracename__dataset_cache_miss_trace.csv, includes indices which have cache miss. Zero stands for the corresponding index has cache hit
      - configurable parameter: (1) cache size: we config cache size based on the number of unique indices. E.g., 0.2 stands for the cache contains 20% of unique indices; (2) block size: column number of blocks. type 0 if only 1 column containing block trace is present; also can be set as average pooling factor (3)data cache trace


    - lru.py: generating cache hit and miss trace with LRU and LFU caching strategy. Time complaxity: O(N) 
      - input: indices trace; 
      - output: (1) tracename_lru_cache.csv, includes an binary array, where 1 stands for cache hit with LRU caching strategy. (2) tracename_lru_miss.csv, includes indices which have cache miss. Zero stands for the corresponding index has cache hit.
        - configurable parameter: (1) cache size: we config cache size based on the number of unique indices. E.g., 0.2 stands for the cache contains 20% of unique indices; (2)data cache trace

 - training caching model: seq2seq_caching.py. Configurable parameters:
   - model input: indices trace; ground-truth: opt cache hit trace (binaray array)
   - configurable parameter: (1)input sequence length N; (2)output sequence length M

 - training prefetching model: train_predition.py, seq2seq_prefetching.py. 
   - model input: indices trace; ground-truth: opt cache miss trace
   - Configurable parameters:
     - input sequence length N, output sequence length M, evaluation window size W
     - loss function: mean_squared_error or IoU loss


## Datasets
  - indices traces: [sythetic dataset for DLRM] (https://github.com/facebookresearch/dlrm_datasets/tree/main/embedding_bag)
  - ground truth generated with optgen: [opt caching, lru, lfu] (https://drive.google.com/drive/folders/140HGV4TZ2IPK1dK2BdYrreCPFeVlmaGq?usp=sharing)

