# cuda-programs

## Prerequistes
1. Install Nvidia Driver
2. Install Nvcc 

## Run
Take mp1 for example. 
```bash
cd mp1_vector_add/
bash compile.sh 
bash bench.sh # check correctness & time for each testcases.
```

Example outout:
```
-------BENCH---------
Case 0
[o] Data size:64, Time spent:39ms

Case 1
[o] Data size:128, Time spent:45ms

Case 2
[o] Data size:56, Time spent:40ms

Case 3
[o] Data size:100, Time spent:39ms

Case 4
[o] Data size:256, Time spent:45ms

Case 5
[o] Data size:130, Time spent:40ms

Case 6
[o] Data size:90, Time spent:48ms

Case 7
[o] Data size:512, Time spent:42ms

Case 8
[o] Data size:90, Time spent:39ms

Case 9
[o] Data size:123, Time spent:45ms
```
