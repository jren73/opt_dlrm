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
import pandas as pd
import torch

def LFU(blocktrace, frame):
        cache = set()
        cache_frequency = defaultdict(int)
        frequency = defaultdict(int)
        
        hit, miss = 0, 0
        lfu = np.zeros(len(blocktrace))
        lfu_miss = np.zeros(len(blocktrace))

        i=0
        for block in tqdm(blocktrace, leave=False):
                frequency[block] += 1
                if block in cache:
                        hit += 1
                        cache_frequency[block] += 1
                        lfu[i] = 1
                
                elif len(cache) < frame:
                        cache.add(block)
                        cache_frequency[block] += 1
                        miss += 1
                        lfu_miss[i] = block

                else:
                        e, f = min(cache_frequency.items(), key=lambda a: a[1])
                        cache_frequency.pop(e)
                        cache.remove(e)
                        cache.add(block)
                        cache_frequency[block] = frequency[block]
                        miss += 1
                        lfu_miss[i] = block
                i = i+1
        
        hitrate = hit / ( hit + miss )
        print(hitrate)

        return lfu,lfu_miss

def LRU(blocktrace, frame):
        
        cache = set()
        recency = deque()
        hit, miss = 0, 0
        lru = np.zeros(len(blocktrace))
        lru_miss = np.zeros(len(blocktrace))
        
        i=0
        for block in tqdm(blocktrace, leave=False):
                
                if block in cache:
                        recency.remove(block)
                        recency.append(block)
                        hit += 1
                        lru[i] = 1
                
                elif len(cache) < frame:
                        cache.add(block)
                        recency.append(block)
                        miss += 1
                        lru_miss[i] = block
                
                else:
                        cache.remove(recency[0])
                        recency.popleft()
                        cache.add(block)
                        recency.append(block)
                        miss += 1
                        lru_miss[i] = block
        i=i+1
        hitrate = hit / (hit + miss)
        print(hitrate)

        return lru,lru_miss

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='caching algorithm.\n')
        parser.add_argument('cache_percent', type=float,  help='relative cache size, e.g., 0.2 stands for 20\% of total trace length\n')
        parser.add_argument('traceFile', type=str,  help='trace file name\n')
        args = parser.parse_args() 

        cache_size = args.cache_percent
        traceFile = args.traceFile


        block_trace, offsets, lengths = torch.load(traceFile)

        block_trace = [x.item() for x in block_trace]
        #block_tmp = []
        #for i in range (10000):
        #        block_tmp.append(block_trace[i].item())

        blockTraceLength = len(block_trace)
        cache_size = int(cache_size * blockTraceLength)
        print("processed!")

        #blockTraceLength = len(block_tmp)
        #cache_size = int(cache_size * blockTraceLength)


        print (f"created block trace list, cache size is {cache_size}")

        # build LRU
        lru_cache,lru_miss = LRU(block_trace, cache_size)
        #lru_cache, lru_miss = LRU(block_tmp, cache_size)
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lru_cache.csv"
        df = pd.DataFrame(lru_cache)
        df.to_csv(cached_trace)
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lru_miss.csv"
        df = pd.DataFrame(lru_miss)
        df.to_csv(cached_trace)

        # build LFU
        lfu_cache, lfu_miss = LFU(block_trace, cache_size)
        #lfu_cache, lfu_miss = LFU(block_tmp, cache_size)
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lfu_cache.csv"
        df = pd.DataFrame(lfu_cache)
        df.to_csv(cached_trace)
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lfu_miss.csv"
        df = pd.DataFrame(lfu_miss)
        df.to_csv(cached_trace)


