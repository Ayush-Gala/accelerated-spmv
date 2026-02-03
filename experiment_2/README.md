# SpMV-CUDA implementation

![Performance Graph - Small Matrices](../static/media/exp2_small.jpg)
![Performance Graph - Large Matrices](../static/media/exp2_large.jpg)

## CUDA Kernel: `coo_spmv_kernel`

```cpp
__global__ void coo_spmv_kernel(int num_nonzeros, const int* __restrict__ rows,
                               const int* __restrict__ cols,
                               const float* __restrict__ vals,
                               const float* __restrict__ x,
                               float* __restrict__ y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_nonzeros) {
        atomicAdd(&y[rows[idx]], vals[idx] * x[cols[idx]]);
    }
}
```

### Kernel Functionality
This kernel performs SpMV by iterating over the non-zero elements and computing their contribution to the output vector `y`. Each thread processes a single non-zero element and updates the corresponding position in `y` using `atomicAdd` to avoid race conditions when multiple threads update the same row.

### Thread Indexing
The global thread ID is computed as follows:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

where:

- `blockIdx.x` is the block index.
- `blockDim.x` is the number of threads per block (set to 256).
- `threadIdx.x` is the thread index within the block.

This indexing method creates a contiguous range of indices from `0` to `(gridSize * blockSize - 1)`, ensuring that each thread processes a unique non-zero element. The `if (idx < num_nonzeros)` check ensures that only valid elements are processed, avoiding out-of-bounds memory access.

## Benchmarking Function: `benchmark_cuda_coo_spmv`
This function handles memory allocation, data transfer, kernel execution, and performance measurement. Key steps include:

1. **Device Memory Allocation:** GPU memory is allocated for all input arrays (`rows`, `cols`, `vals`, `x`, and `y`).
2. **Data Transfer:** The host (CPU) data is copied to device (GPU) memory.
3. **Kernel Launch Configuration:**
   - `blockSize = 256` (256 threads per block).
   - `gridSize = (num_nonzeros + blockSize - 1) / blockSize`, ensuring sufficient threads to cover all non-zero elements.
4. **Performance Measurement:**
   - A warmup run is performed.
   - The kernel is executed multiple times.
   - Performance metrics such as GFLOP/s (floating-point operations per second) and GB/s (memory bandwidth) are computed.

## Correctness Testing
A testing method (enabled with `#ifdef TESTING`) verifies the correctness of CUDA results against a CPU implementation. The maximum absolute difference between the CPU and GPU results is calculated:

```cpp
float max_diff = 0.0f;
for(int i = 0; i < coo.num_rows; i++) {
    float diff = fabsf(y[i] - y_cuda[i]);
    max_diff = max_diff > diff ? max_diff : diff;
}
printf("\nMaximum difference between CPU and CUDA results: %e\n", max_diff);
```
Additionally, diagnostic outputs are written to files for further analysis.

---
---

<br>

# SpMV-CUDA-MPI implementation

## MPI Data Distribution

The code employs a domain decomposition approach to distribute the sparse matrix data, stored in Coordinate (COO) format, across multiple MPI processes.

### Initial Matrix Reading

Only the root process (rank 0) reads the entire COO matrix from the MatrixMarket file and initializes it with random values. The matrix data is then distributed among all processes.

### Distribution Strategy

- The matrix dimensions (rows, columns, and number of nonzero elements) are broadcast to all MPI processes.
- The nonzero elements are divided into approximately equal chunks among all processes.
- Each process receives around `chunk_size = num_nonzeros / world_size` elements.
- If there is a remainder in division, processes with lower ranks receive one extra element to balance the distribution.

### Scatter Implementation

The following code snippet demonstrates how chunk sizes and displacements are computed and how data is scattered:

```c
// Calculate chunk sizes and displacements
int chunk_size = coo.num_nonzeros / world_size;
int remainder = coo.num_nonzeros % world_size;
int* sendcounts = (int*)malloc(world_size * sizeof(int));
int* displs = (int*)malloc(world_size * sizeof(int));

// Each process gets its local portion
MPI_Scatterv(coo.vals, sendcounts, displs, MPI_FLOAT,
             coo.vals, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
```

### Input Vector Sharing

The input vector `x` is broadcast to all processes since each process needs the full vector to compute its local SpMV contribution.

## CUDA Kernel Execution

Each MPI process executes a CUDA kernel to perform SpMV on its assigned portion of the sparse matrix. The CUDA kernel assigns each thread to process a single nonzero entry in the local matrix portion:

### Parallelization Strategy

- Each CUDA thread processes a single nonzero element in the matrix.
- The `atomicAdd` operation ensures correct accumulation when multiple threads update the same output element.
- The kernel is launched with enough threads to cover the local nonzero count: `gridSize = (num_nonzeros + blockSize - 1) / blockSize`.
- Since each process runs the kernel on its local portion and each thread handles one nonzero element, all nonzero elements in the matrix are processed across the entire MPI communicator (`MPI_Comm_World`).

## Result Aggregation and Testing

After local computations, results are combined and verified for correctness. Each process contributes to the final output vector using MPI_Reduce:

```c
MPI_Reduce(y_cuda, global_y, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
```

- All local output vectors are summed element-wise using `MPI_SUM`.
- This operation ensures that partial results from different processes are correctly combined.
- The root process receives the final aggregated result.

### Correctness Testing

The code includes a `#ifdef TESTING` section to validate the correctness of the parallel implementation:

- The maximum absolute difference between CPU and CUDA results is computed.
- The input vector and both output vectors (CPU and CUDA) are written to files for verification.
- This comparison ensures that the distributed CUDA-MPI implementation produces the same results as the sequential CPU version.

---

