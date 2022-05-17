from functools import lru_cache
import sys
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm_notebook as tqdm 
from collections import Counter, deque, defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, confusion_matrix

def LFU(blocktrace, frame):
        cache = set()
        cache_frequency = defaultdict(int)
        frequency = defaultdict(int)
        
        hit, miss = 0, 0
        lfu = np.zeros(blocktrace.size())


        for i, block in tqdm(blocktrace, leave=False):
                frequency[block] += 1
                if block in cache:
                        hit += 1
                        cache_frequency[block] += 1
                        lfu[i] = 1
                
                elif len(cache) < frame:
                        cache.add(block)
                        cache_frequency[block] += 1
                        miss += 1
                        lfu[i] = 0

                else:
                        e, f = min(cache_frequency.items(), key=lambda a: a[1])
                        cache_frequency.pop(e)
                        cache.remove(e)
                        cache.add(block)
                        cache_frequency[block] = frequency[block]
                        miss += 1
                        lfu[i] = 0
        
        hitrate = hit / ( hit + miss )
        print(hitrate)

        return lfu

def LRU(blocktrace, frame):
        
        cache = set()
        recency = deque()
        hit, miss = 0, 0
        lru = np.zeros(blocktrace.size())
        
        for i, block in tqdm(blocktrace, leave=False):
                
                if block in cache:
                        recency.remove(block)
                        recency.append(block)
                        hit += 1
                        lru[i] = 1
                
                elif len(cache) < frame:
                        cache.add(block)
                        recency.append(block)
                        miss += 1
                        lru[i]=0
                
                else:
                        cache.remove(recency[0])
                        recency.popleft()
                        cache.add(block)
                        recency.append(block)
                        miss += 1
                        lru[i]=0
        
        hitrate = hit / (hit + miss)
        print(hitrate)

        return lru

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='caching algorithm.\n')
        parser.add_argument('cache_percent', type=float,  help='relative cache size, e.g., 0.2 stands for 20\% of total trace length\n')
        parser.add_argument('traceFile', type=str,  help='trace file name\n')
        args = parser.parse_args() 

        cache_size = args.cache_percent
        traceFile = args.traceFile


        block_trace, offsets, lengths = torch.load(traceFile)


        blockTraceLength = len(block_trace)
        cache_size = int(cache_size * blockTraceLength)

        print (f"created block trace list, cache size is {cache_size}")

        # build LRU
        lru_cache = LRU(block_trace, cache_size)
        cached_trace = block_trace[block_trace.find("embedding_bag/")+len("embedding_bag/"):s.rfind(".pt")] + "dataset_trace_lru.txt"
        f = open(cached_trace, "w")
        f.write(lru_cache)
        f.close() 

        # build LFU
        lfu_cache = LFU(block_trace, cache_size)
        cached_trace = block_trace[block_trace.find("embedding_bag/")+len("embedding_bag/"):s.rfind(".pt")] + "dataset_trace_lfu.txt"
        f = open(cached_trace, "w")
        f.write(lfu_cache)
        f.close()


