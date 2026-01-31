** NOTE: **

1. All the benchmarks were calculated on node collection 'skylake'. 

<---> <---> <--->

** TEXT: **

Q4. (5 points) Explain the code.

a] spmv-mpi.c 

After looking at the nature of the computation, my approach was to 
1. Broadcast the dimensions of the COO matrix i.e number of rows, columns, and values. (using MPI_Bcast())
2. Broadcast the dense vector x since all nodes will require the entire dense vector. (using MPI_Bcast())
3. Calculate chunk sizes by (num_nonzeros / world_size) and also handling remainders in case the matrix isn't perfectly divisible.
4. Divide the COO matrix into 'world_size' number of chunks and send the ith chunk to the ith node. (Using MPI_Scatterv()).
5. Set the local value of num_nonzeros at each node to chunk size and have each node iterate over their own parts of the chunk.
6. The calculation of num_iterations is done only by the 'root node' which it then broadcasts to all nodes. (using MPI_Bcast()).
7. Once the benchmark is compelete, use MPI_Reduce to collect the local y vectors into a single global_y vector at root node.


b] spmv-omp.c

1. The main thread calls the function benchmark_coo_spmv(). All matrix variables up till this point have a global scope. This means that every thread spawned from the main thread will have access to the COO matrix object.
2. The warmup and main for loops are parallelized using the `#pragma omp parallel for` directive which allows multiple threads to process different nonzero entries in parallel
3. Since multiple threads may update the same row in y, a `#pragma omp atomic` directive ensures atomic uodates. This is important to prevent any race conditions that may occur.
4. The `#pragm omp for` directive has an impicit barrier at the end of each iteration. This ensures that the execution time is recorded when all threads have compeleted the computation.


c] spmv-hybrid.c

1. Similar to the MPI implementation, the root node first reads the matrix dataset and loads it into its local COO object. It also generates the dense vector x.
2. The marix dimensions are broadcasted to all nodes. The dense vector x is also broadcasted since all nodes need it. (using MPI_Bcast())
3. The matrix is then split into chunks based on num_nonzeros / world_size ensuring fair workload. If the matrix isnt perfectly divisible then I handle for remainder values.
4. Each node then receives its chunk of the COO matrix using MPI_Scatterv(). The root node chunks are marked and modified accordingly.
5. Each node then runs the benchmark_coo_spmv() function. I set the local value of num_nonzeros at each node to chunk size and have each node iterate over their own parts of the chunk.
6. The calculation of num_iterations is done only by the 'root node' which it then broadcasts to all nodes. (using MPI_Bcast()).
7. The main Spmv loop is paralellized using `#pragma omp parallel for` directive. Since multiple threads may update the same row in y, a `#pragma omp atomic` directive ensures atomic updates.
8. Once all iterations are complete and the function returns to main, I use `MPI_Reduce` to collect the local y vectors into a single global_y vector at root node.

<---> <---> <--->

Q5. Performance comparisons and thoughts. 

NOTE: Graphs perf-comp.jpg has been attached to the submission. Since the executions times for the first 3 datasets are smaller, their bar plots aren't visible. I have attached an enlarged version of the bar plot for the first 3 datasets undet the file name enlarged-perf-comp.jpg

a] Performance comparison:

1. MPI seems to be the fastest acroll all the datasets, showing the lowest execution times. We can see a significant reduction in execution times for larger matrices like 'Ga3As3H12' and 'pkustk14'. This is likely due to distribution of workload and little to none lock contention during the reduction process.
2. OpenMP shows competitive performance for medium-sized datasets but struggles a bit with larger datasets, possibly because of memory bandwidth limitations and synchronization overhead.
3. The hybrid performances show a tradeoff between the MPI and OpenMP versions. Hybrid-4 performs better than Hybrid-2 and Hybrid-8 indicating that an optimal number of cores can balance the workload and communication overhead effectively.
4. Moving aside from the execution times, if we take a look at the values of GFLOPs and GBYTES for each program, we can see that the MPI program achieves high values of GB per second. This likely indicates that the MPI program uses the memory subsystem efficiently which is critical in memory-bound operations like SpMV. This could also be a result of the communication overhead of the program. Thus if our underlying system cannot support high bandwidths then an MPI program wouldn't be the best choice to make.
5. On the other hand, the OpenMP program consistently maintains higher GFLOP/s values than others. This indicates that the hardware is being utilized efficiently for computation and the algorithm is performing well computationally. This is difficult to achieve for memory-bound computations like that of SpMV.


