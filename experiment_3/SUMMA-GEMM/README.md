# SUMMA-GEMM Implementation

## Table of Contents

- [Description of SUMMA Implementation](#description-of-summa-implementation)
  - [a) Stationary-C](#a-stationary-c)
  - [b) Stationary-A](#b-stationary-a)
  - [c) Stationary-B](#c-stationary-b)
- [Key Design Decisions](#key-design-decisions)
  - [a) Unified Distribution Function with Transpose Flag](#a-unified-distribution-function-with-transpose-flag)
  - [b) Use of MPI_Reduce_Scatter](#b-use-of-mpi_reduce_scatter)
- [Functional Components](#functional-components)
  - [1. Process Grid Creation](#1-process-grid-creation)
  - [2. Data Distribution](#2-data-distribution)
  - [3. Communication](#3-communication)
  - [4. Result Collection](#4-result-collection)
- [Challenges Faced](#challenges-faced)
  - [1. Understanding Stationary-A and Stationary-B Implementations](#1-understanding-stationary-a-and-stationary-b-implementations)
  - [2. Finding Appropriate MPI Directives](#2-finding-appropriate-mpi-directives)
- [Testing](#testing)
  - [Testing Environment](#testing-environment)
  - [Test Cases](#test-cases)
    - [1. Square Matrices](#1-square-matrices)
    - [2. Rectangular Matrices](#2-rectangular-matrices)
  - [Correctness Measures](#correctness-measures)
  - [Performance Measures](#performance-measures)

---

## Description of SUMMA Implementation

The SUMMA algorithm partitions matrices A, B, and C into blocks that are distributed across a 2D process grid. The algorithm then proceeds in stages, with each stage involving communication to align the required matrix blocks for local computation. The three variants - Stationary-A, Stationary-B, and Stationary-C - differ in which matrix remains stationary during computation.

### a) Stationary-C

In the Stationary-C variant, matrix C remains stationary throughout the computation process. Each process maintains its local block of C (C_local) and accumulates results into it. The algorithm works through a sequence of stages equal to the grid size:

- For each stage s, processes that own the corresponding blocks of A and B copy them to temporary buffers (A_temp and B_temp).
- A_temp is broadcast along each row, while B_temp is broadcast along each column.
- Each process performs local matrix multiplication: C_local += A_temp * B_temp
- This continues until all required blocks have been processed across all stages.

The key characteristic is that C_local accumulates contributions from multiple A and B block combinations, with C blocks fixed at their owner processes.

### b) Stationary-A

In the Stationary-A variant, matrix A remains stationary at its assigned process. This implementation uses a transposed distribution of matrix B to optimize communication patterns:

- Matrix A is distributed normally, but matrix B is distributed in a transposed manner (process (i,j) receives B block (j,i)).
- For each stage s, only B blocks need to be broadcast along columns.
- Each process calculates C_temp = A_local * B_temp using its stationary A_local block and the received B_temp.
- An MPI_Reduce_scatter operation aggregates C_temp values to the appropriate processes in the row that should hold the final C values.

This approach reduces communication volume as A blocks remain fixed, requiring only B blocks to be communicated during computation.

### c) Stationary-B

In the Stationary-B variant, matrix B remains stationary at its assigned process. Similar to Stationary-A but with inverted roles:

- Matrix A is distributed in a transposed manner, while matrix B is distributed normally.
- For each stage s, only A blocks need to be broadcast along rows.
- Each process calculates C_temp = A_temp * B_local using the received A_temp and its stationary B_local block.
- MPI_Reduce_scatter operation aggregates C_temp values to the appropriate processes in the column that should hold the final C values.

This variant is particularly beneficial for matrices where B is more frequently referenced or where keeping B stationary offers memory or computation advantages.

## Key Design Decisions

### a) Unified Distribution Function with Transpose Flag

A significant design choice was implementing a single distribute_matrix_blocks function that handles both normal and transposed distribution patterns:

```c
void distribute_matrix_blocks(float *global_matrix, float *local_block, int m,
   int n, int local_m, int local_n, MPI_Comm comm_2d,
   int grid_size, int rank, int transpose)
```

The transpose parameter determines whether blocks are distributed in the standard manner (process (i,j) gets block (i,j)) or transposed manner (process (i,j) gets block (j,i)). This is implemented through the displacement calculation:

```c
if(transpose){
  for (int i = 0; i < grid_size; i++) {
    for (int j = 0; j < grid_size; j++) {
      // In the transposed case, we send to (j,i) instead of (i,j)
      send_counts[j * grid_size + i] = 1;
      displs[j * grid_size + i] = i * local_m * n + j * local_n;
    }
  }
}
else{
  for (int i = 0; i < grid_size; i++) {
    for (int j = 0; j < grid_size; j++) {
      send_counts[i * grid_size + j] = 1;
      displs[i * grid_size + j] = i * local_m * n + j * local_n;
    }
  }
}
```

This unified approach significantly reduced code redundancy and simplified maintenance, while providing the flexibility needed for different algorithm variants.

### b) Use of MPI_Reduce_Scatter

The implementation uses MPI_Reduce_scatter for the Stationary-A and Stationary-B variants instead of other potential MPI collective operations:

```c
MPI_Reduce_scatter(C_temp, C_local, recvcounts, MPI_FLOAT, MPI_SUM, row_comm);
```

This operation combines reduction (summing partial results) with scattering (distributing results to their final destination processes) in a single communication step. The benefits include:

- Communication efficiency by combining 2 operations (reduce and scatter) into 1
- Reduced memory footprint as intermediate buffers for separate reduce and scatter operations are avoided.
- Optimized data movement that aligns with the SUMMA's computational pattern.

The recvcounts array is dynamically configured to ensure results are delivered to the correct processes based on the current stage.

## Functional Components

### 1. Process Grid Creation

The implementation creates a 2D Cartesian process grid using MPI's topology functionality:

```c
int grid_size = (int)sqrt(nprocs);
int dims[2] = {grid_size, grid_size};
int periods[2] = {0, 0};  // Non-periodic grid
int coords[2];
MPI_Comm grid_comm, row_comm, col_comm;

// Creating cartesian communicator
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

// Getting process coordinates
MPI_Cart_coords(grid_comm, rank, 2, coords);

// Recording the row and column of the process in the grid
int proc_row = coords[0];
int proc_col = coords[1];

// Create row communicators (all processes in the same row)
MPI_Comm_split(grid_comm, proc_row, proc_col, &row_comm);
  
// Create column communicators (all processes in the same column)
MPI_Comm_split(grid_comm, proc_col, proc_row, &col_comm);
```

This creates a grid_size × grid_size process grid where each process is identified by its row and column coordinates. The implementation further establishes specialized communicators for row and column operations. These specialized communicators enable efficient broadcast operations along rows and columns, which is critical for the SUMMA algorithm's performance.

### 2. Data Distribution

The distribution of matrices uses MPI's subarray functionality to define complex data types that represent matrix blocks:

```c
MPI_Datatype block_type, resized_block_type;
int starts[2] = {0, 0};
int subsizes[2] = {local_m, local_n};
int bigsizes[2] = {m, n};

MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &block_type);
MPI_Type_commit(&block_type);

MPI_Type_create_resized(block_type, 0, sizeof(float), &resized_block_type);
MPI_Type_commit(&resized_block_type);

MPI_Scatterv(global_matrix, send_counts, displs, resized_block_type, 
             local_block, local_m * local_n, MPI_FLOAT, 0, comm_2d);
```

The resized datatype ensures proper memory addressing when accessing blocks. The MPI_Scatterv operation then distributes blocks to processes. For Stationary-C, both matrices A and B are distributed normally i.e. each process receives blocks directly corresponding to its grid position. For Stationary-A, matrix A is distributed normally, but matrix B is transposed, while for Stationary-B, the distribution pattern is reversed - A is transposed while B is normal. The transpose flag ensures process (i,j) receives B block (j,i), which optimizes for the subsequent computation pattern.

### 3. Communication

All variants use MPI collective operations for communication. Time measurements track communication overhead. Stationary-C requires two broadcasts per stage which ensures all processes have the A and B blocks needed for the current stage's computation.

```c
// Broadcast A_temp along the row
MPI_Bcast(A_temp, local_m * local_k, MPI_FLOAT, A_owner_col, row_comm);

// Broadcast B_temp along the column
MPI_Bcast(B_temp, local_k * local_n, MPI_FLOAT, B_owner_row, col_comm);
```

Both Stationary-A and Stationary-B perform one broadcast and one reduce-scatter per stage. Stationary-A broadcast distributes B blocks, while reduce-scatter consolidates partial results to their target processes. For stationary-B, reduce-scatter operates on column communicators rather than row communicators as in Stationary-A.

**Example for Stationary-A:**

```c
// Broadcast B_temp along the column
MPI_Bcast(B_temp, local_k * local_n, MPI_FLOAT, B_owner_row, col_comm);

// Reduce-scatter C_temp values to appropriate processes
MPI_Reduce_scatter(C_temp, C_local, recvcounts, MPI_FLOAT, MPI_SUM, row_comm);
```

### 4. Result Collection

All variants use identical result collection logic through MPI_Gatherv. This operation collects local blocks of C from all processes and assembles them into the complete result matrix at rank 0. The custom datatype resized_block_type enables proper placement of blocks within the global matrix structure.

```c
MPI_Gatherv(C_local, local_m * local_n, MPI_FLOAT,
            C, recv_counts, displs, resized_block_type,
            0, grid_comm);
```

## Challenges Faced

### 1. Understanding Stationary-A and Stationary-B Implementations

One major challenge was correctly understanding the Stationary-A and Stationary-B variants. The default implementation of Stationary-A computes C = AB^T, while Stationary-B computes C = A^TB. Several research papers were studied to gain clarity on the proper implementation of these variants. The key insight was that these variants require different matrix distribution patterns. For Stationary-A, matrix B needs to be distributed in a transposed manner, while for Stationary-B, matrix A requires the transposed distribution. This insight led to the implementation of the transpose flag in the distribution function.

### 2. Finding Appropriate MPI Directives

Identifying the optimal MPI directive for the reduce-scatter operation required extensive documentation research. The MPI_Reduce_scatter collective operation was selected as it precisely matches the algorithmic pattern needed. This operation not only combines partial results but also distributes them to their target processes in a single step, avoiding the need for separate reduce and scatter operations that would increase communication overhead.

Additional complexities included correctly setting up the recvcounts array to ensure partial results were sent to the appropriate processes based on the current stage and algorithm variant. The implementation includes detailed timing measurements that allow for performance analysis of different phases of the algorithm, including the communication overhead, which is critical for identifying optimization opportunities in large-scale parallel computing environments.

## Testing

### Testing Environment

The testing was conducted on the ARC cluster at NCSU, utilizing AMD Epyc Rome processors across nodes c[20-25,33,37-48,73-79] through the Rome queue. The hardware configuration consisted of:

- Tyan S8021 Motherboard with IPMI 2.0 PCIe 3
- Transport HX GA88-B8021 (B8021G88V2HR-2T-RM-N) with 4 GPU slots + 1 half-size PCI-E slot (except for node c33)
- 128GB DDR4 3200 ECC DRAM per node

### Test Cases

#### 1. Square Matrices

For square matrices, we conducted a systematic evaluation using matrices of dimensions 4096×4096, 8192×8192, 16384×16384, 32768×32768, and 64k×64k. These were distributed across process grids of sizes 2×2 (4 processes), 4×4 (16 processes), and 8×8 (64 processes) to evaluate scaling behavior.

The primary metrics of interest for square matrices were:

- Scaling efficiency as the number of processes increases
- Load balancing characteristics across the process grid
- Communication overhead in the broadcast operations (measured via broadcast_sum timings)
- Comparative performance between the three SUMMA variants (Stationary-A, Stationary-B, and Stationary-C)
- Memory utilization patterns across different grid configurations
- Relative performance of parallel implementation versus serial execution (captured in the timing outputs)

The square matrix tests provide insight into how each algorithm variant handles evenly distributed computational loads and the efficiency of the block distribution strategy.

#### 2. Rectangular Matrices

For rectangular matrices, we focused on tall-skinny configurations where n=128 while maintaining the same m and k dimensions as the square matrices. These asymmetric matrices were tested across the same process grid configurations (2×2, 4×4, and 8×8).

The focus points for rectangular matrix evaluation were:

- Efficiency of load distribution for asymmetric workloads
- Communication patterns when matrix dimensions don't align with process grid dimensions
- Performance characteristics of the three SUMMA variants when dealing with non-square matrices
- Identification of potential bottlenecks in the algorithm when handling irregular shapes
- Memory access patterns and their impact on overall performance

These tests are crucial for understanding real-world application scenarios where input matrices often have non-square dimensions, revealing how the different stationary approaches adapt to uneven computational distributions.

### Correctness Measures

- The verify_result() function performs an element-by-element comparison between the parallel computed result and a serial computation baseline.
- Extensive debugging infrastructure is included in the code through commented-out print statements that can be enabled to visualize:
  - Initial matrix distribution across processes
  - Process grid creation and communicator establishment
  - Local block contents after distribution
  - Intermediate computation results at each iteration
  - Final local results before gathering
- Each process maintains its 2D coordinates in the Cartesian grid to ensure proper data routing during the broadcast and reduce operations.
- MPI barriers (MPI_Barrier()) are strategically placed to maintain synchronization and prevent race conditions during the algorithm execution.

### Performance Measures

Performance measurement was implemented using high-precision MPI timing functions and phase-specific counters.

**MPI_Wtime() calls capture timestamps at critical sections:**

- `parallel_start_time`: Beginning of parallel execution
- `blk_dist_end_time`: Completion of initial block distribution
- `main_comp_end_time`: End of the main computational loop
- `parallel_end_time`: Completion of result gathering

**Dedicated communication timing:**

- `broadcast_start` and `broadcast_end`: Measure the time spent in MPI broadcast operations.
- `broadcast_sum`: Accumulates the total communication overhead during the main computation phase.

**Performance metrics generated for each run:**

- Initial block distribution time
- Main computation execution time
- Total messaging time in the main computation
- MPI_Gatherv execution time
- Total parallel computation time
- Serial matrix multiplication time for comparison

The granular timing approach enables detailed analysis of where time is spent in each algorithm variant, facilitating identification of which variant is most efficient for specific matrix dimensions and process grid configurations.
