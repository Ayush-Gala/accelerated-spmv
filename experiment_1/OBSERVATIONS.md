# Experiment 1 Observations

> **Note:** All benchmarks were calculated on the 'Skylake' node collection.

---

## Implementation Details

### MPI Implementation (`spmv-mpi.c`)

After analyzing the computation, the approach was:

1. **Broadcast matrix dimensions** (rows, columns, nonzeros) using `MPI_Bcast()`
2. **Broadcast dense vector x** to all nodes since all need the entire vector
3. **Calculate chunk sizes** by `num_nonzeros / world_size`, handling remainders for non-divisible matrices
4. **Distribute COO matrix** into `world_size` chunks, sending the i-th chunk to the i-th node using `MPI_Scatterv()`
5. **Set local num_nonzeros** at each node to chunk size for local iteration
6. **Calculate num_iterations** only at root node, then broadcast to all nodes using `MPI_Bcast()`
7. **Collect results** using `MPI_Reduce` to gather local y vectors into a single global_y vector at root node

### OpenMP Implementation (`spmv-omp.c`)

1. The main thread calls `benchmark_coo_spmv()`. All matrix variables have global scope, so every spawned thread can access the COO matrix object
2. Warmup and main loops are parallelized using `#pragma omp parallel for` directive, allowing multiple threads to process different nonzero entries in parallel
3. Since multiple threads may update the same row in y, `#pragma omp atomic` directive ensures atomic updates, preventing race conditions
4. The `#pragma omp for` directive has an implicit barrier at the end of each iteration, ensuring execution time is recorded when all threads complete

### Hybrid Implementation (`spmv-hybrid.c`)

1. Similar to MPI, the root node reads the matrix dataset into its local COO object and generates dense vector x
2. **Broadcast matrix dimensions** and dense vector x to all nodes using `MPI_Bcast()`
3. **Split matrix** into chunks based on `num_nonzeros / world_size`, handling remainders
4. **Distribute chunks** to each node using `MPI_Scatterv()`, with root node chunks marked and modified accordingly
5. Each node runs `benchmark_coo_spmv()` with local `num_nonzeros` set to chunk size
6. **Calculate num_iterations** only at root node, then broadcast using `MPI_Bcast()`
7. The main SpMV loop is parallelized using `#pragma omp parallel for` directive. `#pragma omp atomic` ensures atomic updates for thread safety
8. **Collect results** using `MPI_Reduce` to gather local y vectors into a single global_y vector at root node

---

## Performance Analysis

### Key Findings

1. **MPI Performance**  
   MPI achieves the lowest execution times across all datasets, with significant reductions for larger matrices like `Ga3As3H12` and `pkustk14`. This is due to efficient workload distribution and minimal lock contention during reduction.

2. **OpenMP Performance**  
   OpenMP shows competitive performance for medium-sized datasets but struggles with larger ones, likely due to memory bandwidth limitations and synchronization overhead.

3. **Hybrid Performance**  
   Hybrid configurations show a tradeoff between MPI and OpenMP. Hybrid-4 (N=4) performs better than Hybrid-2 and Hybrid-8, indicating an optimal core count balances workload and communication overhead.

4. **Memory Efficiency (GB/s)**  
   MPI achieves high GB/s values, indicating efficient memory subsystem utilization—critical for memory-bound operations like SpMV. However, this may also reflect communication overhead, so high bandwidth support is essential for MPI effectiveness.

5. **Computational Efficiency (GFLOP/s)**  
   OpenMP consistently maintains higher GFLOP/s values, indicating efficient hardware utilization for computation. This is notable for memory-bound computations like SpMV.
