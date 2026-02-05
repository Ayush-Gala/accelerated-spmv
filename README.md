# Accelerated Sparse Matrix Operations with Parallel Computing

![MPI](https://img.shields.io/badge/MPI-OpenMPI%20%7C%20MVAPICH2-blue)
![OpenMP](https://img.shields.io/badge/OpenMP-5.0-green)
![CUDA](https://img.shields.io/badge/CUDA-11.x-brightgreen)
![C](https://img.shields.io/badge/C-99-orange)
![C++](https://img.shields.io/badge/C++-14-orange)
![Slurm](https://img.shields.io/badge/Scheduler-Slurm-red)
![PBS](https://img.shields.io/badge/Scheduler-PBS-red)
![GCC](https://img.shields.io/badge/Compiler-GCC-yellow)
![NVCC](https://img.shields.io/badge/Compiler-NVCC-green)
![InfiniBand](https://img.shields.io/badge/Network-InfiniBand%20HDR-purple)
![GPFS](https://img.shields.io/badge/Storage-GPFS-lightgrey)
![BeeGFS](https://img.shields.io/badge/Storage-BeeGFS-lightgrey)
![Linux](https://img.shields.io/badge/OS-Rocky%20Linux%208-blue)

## Overview

This project explores high-performance parallel computing techniques for accelerating sparse matrix multiplication operations which are critical kernels in scientific computing, machine learning, and data analytics. Through three comprehensive experiments, I implemented and benchmarked various parallelization strategies across shared-memory, distributed-memory, and GPU-accelerated architectures.

### Problem Statement

Sparse Matrix-Vector Multiplication (SpMV) and Matrix-Matrix Multiplication (GEMM) are fundamental operations in:
- High-performance scientific computing applications
- Deep learning inference with pruned neural networks
- Graph analytics and network analysis
- Computational fluid dynamics and structural engineering simulations

These operations present unique challenges due to irregular memory access patterns, load imbalancing, and the need to efficiently handle sparsity while maximizing hardware utilization.

## Table of Contents

- [Project Structure](#project-structure)
- [Experiments Overview](#experiments-overview)
  - [Experiment 1: Multi-Paradigm CPU Parallelization](#experiment-1-multi-paradigm-cpu-parallelization)
  - [Experiment 2: GPU-Accelerated Computing](#experiment-2-gpu-accelerated-computing)
  - [Experiment 3: Scalable Matrix Multiplication (SUMMA Algorithm)](#experiment-3-scalable-matrix-multiplication-summa-algorithm)
- [Key Observations](#key-observations)
- [Testing Methodology & Experimental Setup](#testing-methodology--experimental-setup)
- [Future Work](#future-work)
- [Tools & Technologies](#tools--technologies)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Project Structure

```
accelerated-spmv/
├── experiment_1/          # CPU-based SpMV: OpenMP, MPI, Hybrid (MPI+OpenMP)
├── experiment_2/          # GPU-accelerated SpMV: CUDA and MPI+CUDA
├── experiment_3/          # Dense GEMM using SUMMA algorithm with MPI
├── example_matrices/      # Sample test matrices in MatrixMarket format
├── include/               # Shared header files for matrix I/O and utilities
└── static/                # Performance visualizations and documentation
```

## Experiments Overview

### Experiment 1: Multi-Paradigm CPU Parallelization

**Objective:** Compare three parallelization approaches for SpMV on CPU architectures.

**Implementations:**
- **OpenMP:** Shared-memory parallelization using thread-level parallelism with atomic operations
- **MPI:** Distributed-memory parallelization with data decomposition and collective communication
- **Hybrid (MPI+OpenMP):** Nested parallelism combining MPI process distribution with OpenMP threading

**Key Findings:**
- MPI achieved lowest execution times through efficient workload distribution and minimal lock contention
- OpenMP demonstrated highest GFLOP/s rates but encountered memory bandwidth bottlenecks on large matrices
- Hybrid approaches revealed optimal configurations balancing inter-node communication with intra-node threading
- Performance analysis identified memory-bound vs. compute-bound regimes for different matrix sizes

### Experiment 2: GPU-Accelerated Computing

**Objective:** Leverage GPU parallelism and explore distributed GPU computing for SpMV.

**Implementations:**
- **CUDA:** Single-GPU implementation with optimized kernel launch configurations
- **MPI+CUDA (Multi-GPU):** Distributed approach across 2 and 4 GPU nodes with MPI coordination

**Key Findings:**
- CUDA implementations outperformed all CPU variants by 7-50× depending on matrix structure
- 4-node MPI+CUDA configuration achieved best performance with near-linear scaling to 4 GPUs
- Atomic contention identified as performance bottleneck for power-law graphs with high-degree vertices
- Memory-bound nature of SpMV highlighted importance of bandwidth optimization over raw compute power

**Technical Challenges Addressed:**
- Atomic operation contention in sparse accumulation patterns
- Multi-GPU device selection and workload distribution
- Host-device memory transfer optimization

### Experiment 3: Scalable Matrix Multiplication (SUMMA Algorithm)

**Objective:** Implement the communication-optimal SUMMA algorithm for dense matrix multiplication across 2D processor grids.

**Implementations:**
- **Stationary-A:** A matrix blocks remain stationary, B blocks broadcast, partial C results reduced
- **Stationary-B:** B matrix blocks remain stationary, A blocks broadcast, partial C results reduced
- **Stationary-C:** Both A and B blocks broadcast, C blocks remain stationary

**Key Findings:**
- Achieved speedups exceeding 500× on 8×8 processor grids for large matrices (16K×16K)
- Stationary-C provided balanced performance across matrix shapes with symmetric communication
- Stationary-A excelled for tall-skinny matrices (common in ML applications) with 2.43× faster communication
- Communication overhead grew from 0.07% (2×2 grid) to 49% (8×8 grid), revealing scalability limits
- Strong scaling demonstrated up to 64 processes with diminishing returns beyond optimal grid configurations

**Communication Analysis:**
- Message count complexity: O(P²) broadcasts across P iterations for all variants
- Data movement patterns optimized for rectangular matrix structures
- Reduction operations strategically placed to minimize synchronization overhead

## Testing Methodology & Experimental Setup

Rigorous testing and validation were essential components of this project. All implementations underwent comprehensive correctness validation and performance benchmarking using consistent methodologies across experiments.

### Benchmarking Methodology

**Warmup Iterations:**
Every benchmarking run included an initial warmup iteration executed before actual measurement began. This warmup phase was critical to eliminate cold-start effects:
- Populated CPU caches with frequently accessed data structures
- Allowed branch predictors to learn execution patterns
- Triggered NUMA page allocation and memory binding on multi-socket systems
- Stabilized GPU kernel launch overhead and memory allocation states

Without warmup iterations, initial measurements showed up to 2× variance compared to steady-state performance, making meaningful comparison impossible.

**Multiple Independent Runs:**
Each configuration was executed **5 times independently** to account for system noise and variance. Reported metrics represent the **average across all runs** with outliers (e.g., interrupted by system processes) manually identified and excluded. For critical comparisons, I also computed standard deviation to ensure variance remained within acceptable bounds (<5% coefficient of variation).

**Iteration Count Determination:**
Rather than using fixed iteration counts, I implemented adaptive iteration counting based on a configurable `TIME_LIMIT` parameter (defined in `config.h`). The system:
1. Estimated single-iteration execution time using a trial run
2. Calculated required iterations to reach the time limit (minimum 10 iterations)
3. Executed the benchmark for that many iterations
4. Averaged timing results across iterations

This approach ensured sufficient sampling for fast operations (thousands of iterations for small matrices) while keeping total runtime reasonable for expensive operations (tens of iterations for large multi-GPU runs).

**Performance Metrics:**
Each benchmark reported multiple complementary metrics to capture different performance aspects:

- **Execution Time (ms):** Wall-clock time per iteration, measured using high-resolution timers
- **GFLOP/s:** Computational throughput calculated as `(2 × num_nonzeros × iterations) / (time_seconds × 10^9)` for SpMV (multiply-add per nonzero = 2 FLOPs)
- **GB/s:** Memory bandwidth utilization calculated based on bytes accessed (matrix storage + vector reads/writes)
- **Speedup:** Ratio of sequential baseline time to parallel implementation time
- **Efficiency:** Speedup divided by number of processing units (cores/processes), indicating utilization quality

**Timing Precision:**
- **CPU implementations:** Used `clock_gettime(CLOCK_MONOTONIC)` for nanosecond-resolution timing on Linux
- **GPU implementations:** Used CUDA events (`cudaEventRecord()` and `cudaEventElapsedTime()`) for accurate kernel timing, excluding host-device transfer overhead when explicitly analyzing computation performance
- **Cycle-accurate measurements:** Employed RDTSC (Read Time-Stamp Counter) for micro-benchmarking critical code sections

### Correctness Validation

**Automated Testing Framework:**
All parallel implementations included compile-time testing modes activated with the `-D TESTING` flag during compilation. In testing mode, implementations performed additional validation:

1. **Sequential Reference Computation:**
   - Generated a "ground truth" output by computing SpMV sequentially on the CPU
   - Stored reference results in `seq_y` vector

2. **Parallel Implementation Execution:**
   - Ran the parallel implementation (OpenMP/MPI/CUDA/Hybrid)
   - Stored parallel results in separate output vector (`test_y` or `y_cuda`)

3. **Numerical Comparison:**
   - Computed maximum absolute difference: `max_diff = max(|seq_y[i] - parallel_y[i]|)` across all elements
   - Reported maximum difference to identify any discrepancies
   - Used floating-point epsilon-based thresholds to account for acceptable rounding differences

4. **Diagnostic File Output:**
   - Wrote input vector `x` to file for manual inspection
   - Wrote sequential output `seq_y` to file
   - Wrote parallel output `test_y`/`y_cuda` to file
   - Enabled external verification using the `compare.sh` script

**Validation Across Distributed Systems:**
For MPI and MPI+CUDA implementations, validation was more complex due to data distribution:
- Only rank 0 performed the sequential reference computation
- Each rank computed its local portion in parallel
- `MPI_Reduce` with `MPI_SUM` operation aggregated partial results at rank 0
- Final comparison occurred only at rank 0 after global reduction
- Ensured partial results summed correctly despite distributed computation

**Matrix Test Suite:**
Testing utilized diverse matrices from the SuiteSparse Matrix Collection to validate correctness across different sparsity patterns:

| Matrix | Rows | Cols | Nonzeros | Characteristics |
|--------|------|------|----------|----------------|
| `bfly_G_10.mtx` | 10,240 | 10,240 | 98,304 | Butterfly graph topology |
| `D6-6.mtx` | 9,216 | 9,216 | 73,728 | Structured sparse matrix |
| `dictionary28.mtx` | 52,652 | 52,652 | 89,164 | Dictionary graph (sparse) |
| `Ga3As3H12.mtx` | 61,349 | 61,349 | 2,982,788 | Quantum chemistry (dense sparse) |
| `pkustk14.mtx` | 151,926 | 151,926 | 7,391,792 | Structural engineering |
| `roadNet-CA.mtx` | 1,965,206 | 1,965,206 | 2,766,607 | California road network (power-law) |

This diverse test suite ensured implementations handled various sparsity ratios (0.01% to 48%), matrix sizes (9K to 2M rows), and structural patterns (graphs, grids, scientific problems).

### Experimental Setup

### Hardware Configuration

**CPU Nodes (NCSU ARC - Skylake Queue):**
- **Processor:** Intel Xeon Skylake Silver
- **Architecture:** x86_64 with AVX-512 support
- **Memory:** 96GB DDR4 2666 MHz ECC DRAM
- **Storage:** Intel DC S4500 240GB SSD + local scratch space
- **Nodes Used:** c[4-19,26] from Skylake partition

**CPU Nodes (NCSU Hazel - Rome Queue):**
- **Processor:** AMD EPYC 7302P (16-core, 3.0 GHz base)
- **Architecture:** x86_64 Rome microarchitecture
- **Memory:** 128GB DDR4 3200 MHz ECC DRAM
- **Motherboard:** Tyan S8021 with IPMI 2.0 PCIe 3.0
- **Nodes Used:** c[20-25,33,37-48,57,73-79] from Rome partition (`-p rome`)

**GPU Nodes (NCSU ARC - GPU Queue):**
- **GPU:** NVIDIA Tesla/Volta architecture
- **CUDA Cores:** Thousands of parallel CUDA cores per device
- **GPU Memory:** High-bandwidth GDDR memory
- **Configurations Tested:** Single GPU, 2-node multi-GPU, 4-node multi-GPU

**Network Infrastructure (Hazel Cluster):**
- **Topology:** InfiniBand HDR (High Data Rate) network
- **Core Switches:** Mellanox QM8700 HDR 200 Gbps (configured as 80-port 100 Gbps)
- **Node Adapters:** Mellanox ConnectX-6 HDR 100 Gbps NICs
- **MPI Communication:** Dedicated InfiniBand network for low-latency message passing
- **Management Network:** GigE switches for system administration

**Storage Infrastructure:**
- **Parallel File System:** GPFS (General Parallel File System) - scratch directories for active job data
  - Connected via multiple 100 Gbps links to HPC network
  - Lenovo DSS storage arrays with automatic 30-day file cleanup
  - Optimized for high-throughput parallel I/O operations
- **Large Data Storage:** BeeGFS distributed file system for datasets and long-term storage
- **Home Directories:** NFS-mounted with 40GB quota per user (code and small files)

**Processor Grids (SUMMA Experiments):**
- Tested configurations: 2×2, 4×4, 8×8 process grids (up to 64 MPI processes)
- Cartesian topology mapping for optimal communication patterns

### Software Environment

**Operating System:** Rocky Linux 8.x (OpenHPC 2.x distribution) with 4.18.x kernel

**Compiler Toolchain:**
- GCC 8.x/9.x with optimization flags (`-O3`, `-march=native`)
- NVCC (NVIDIA CUDA Compiler) with compute capability targeting
- NVHPC/PGI compilers for additional GPU optimization

**MPI Implementations:**
- OpenMPI 4.x with InfiniBand Verbs support
- MVAPICH2 (high-performance MPI over InfiniBand)

**Parallel Programming Libraries:**
- OpenMP (via GCC and NVHPC compilers)
- CUDA Toolkit 11.x for GPU programming

**File Systems & I/O:**
- GPFS for scratch storage (high-performance parallel I/O during job execution)
- BeeGFS for large dataset storage (distributed file system)
- NFS for home directories (quota-managed, 40GB limit)

**Job Scheduling:**
- Slurm Workload Manager with partition-based resource allocation (`#SBATCH`)
- PBS Pro for job arrays and dependencies (`#PBS`)
- Queue management: skylake, rome, gpu partitions

**Performance Measurement:**
- High-resolution timers using `clock_gettime()` and RDTSC
- CUDA events for GPU kernel timing
- Custom timing utilities with microsecond precision

## Key Observations

Through the course of implementing and benchmarking these parallel algorithms, I gained several critical insights that fundamentally shaped my understanding of high-performance computing.

### Architecture-Specific Optimization is Non-Negotiable

The most striking realization was how dramatically performance characteristics change across different hardware platforms. When I first implemented SpMV with OpenMP, I was impressed by the high GFLOP/s numbers which meant that my parallel threads were doing excellent computational work. But when I profiled memory bandwidth utilization, I discovered the real bottleneck: the memory subsystem couldn't keep up with my threads. This was a humbling lesson that raw computational power means little if you're starved for data.

Conversely, with MPI across distributed nodes, I initially worried about communication overhead. Yet for larger matrices, MPI consistently outperformed my carefully optimized OpenMP implementation. MPI's distributed memory architecture meant each process had its own memory bandwidth, effectively multiplying available bandwidth linearly with process count.

The GPU experiments. The 7-50× speedups weren't just about having thousands of CUDA cores; they came from understanding that GPUs excel at bandwidth-bound operations when you can saturate their memory buses with parallel requests. But this advantage evaporated for power-law graphs where atomic contention serialized operations. The hardware giveth, and the sparsity pattern taketh away.

### Communication is Often the Real Enemy

My SUMMA implementation taught me that elegant algorithms on paper can become communication nightmares at scale. Watching the communication-to-computation ratio climb from 0.07% to 49% as I scaled from a 2×2 to an 8×8 was interesting. At small scales, the kernel dominated execution time exactly as expected. But at 64 processes, I spent more time moving data between nodes than actually computing.

This forced me to deeply understand the difference between the three SUMMA variants. Stationary-C felt intuitive. Broadcast everything, compute locally. But for tall-skinny matrices, those B-block broadcasts were killing performance. Switching to Stationary-A and reducing broadcast volume by 50% led to immediate performance gains. The lesson: minimizing data movement is just as important as minimizing FLOPs, sometimes more so.

What surprised me most was how network topology mattered. InfiniBand's low latency masked inefficiencies at small scales, but those inefficiencies compounded at larger scales. Understanding when to use `MPI_Bcast` versus `MPI_Scatter`, when to overlap communication with computation, when to accept redundant computation to avoid communication. They're the difference between linear scaling and hitting a wall at 16 processes.

### The Hybrid Complexity Tax is Real

I approached hybrid MPI+OpenMP programming with optimism. "Why not get the best of both worlds?" My Hybrid-4 configuration (4 MPI processes, 2 threads each) did find a sweet spot, but Hybrid-8 performed worse than pure MPI despite using the same total core count. Thread synchronization overhead combined with MPI message passing created double the synchronization points.

### Sparsity Destroys Regularity

Dense matrix operations are predictable. Every thread does useful work, memory accesses are regular, and performance models are straightforward. Sparse operations shattered all these assumptions. In my COO SpMV implementation, multiple threads would often try updating the same output vector element, forcing atomic operations. For structured matrices this was manageable, but for power-law graphs with high-degree vertices, contention became severe.

I measured one case where a single highly connected vertex caused thousands of atomic serializations, creating a bottleneck that CPU utilization graphs couldn't even capture. This taught me that analyzing sparse algorithms requires understanding the actual sparsity pattern, not just the sparsity ratio. A 99% sparse matrix can perform completely differently depending on whether those nonzeros are uniformly distributed or clustered.

### Performance Measurement is an Art

Early in the project, I thought timing code was straightforward: wrap a timer around the computation, divide by iterations, done. I quickly learned this was naive. My first OpenMP runs showed wild variance across trials, sometimes 2× differences. The culprit? I wasn't doing warmup iterations. Cold cache, branch predictors learning patterns, NUMA page allocation, etc. all these effects contaminated my first measurement.

I also learned that choosing what to measure matters as much as how to measure it. For MPI implementations, should I time just the computation kernel, or include `MPI_Reduce`? The answer depends on what question you're asking. For algorithmic comparison, exclude it. For end-to-end performance, include it. In my experiments, I generate multiple metrics (time, GFLOP/s, GB/s, speedup, efficiency) because each revealed different bottlenecks.

The GPU work added another layer: `cudaEventRecord()` times kernel execution, but misses host-device transfer time. For small matrices, transfers dominated total runtime, making kernel optimization pointless. This taught me to profile the entire workflow, not just the "interesting" computation part.

### Scalability is a Curve, Not a Line

My undergraduate understanding of parallel computing suggested: "Add more processors, get proportionally faster." The SUMMA experiments definitively proved this wrong. I saw beautiful near-linear scaling from 1 to 4 processes, then good scaling to 16, then diminishing returns to 64. This wasn't a implementation bug, it was fundamental. Amdahl's Law is a physical constraint you hit hard at scale.

More interestingly, the optimal configuration varied by problem size. For 4K×4K matrices, an 8×8 grid was overkill and the communication overhead wasn't worth it. For 16K×16K matrices, 8×8 finally made sense. This taught me that scalability isn't a property of an algorithm but a relationship between algorithm, problem size, and hardware characteristics.

### What I'd Do Differently

If I could restart this project with my current knowledge, I'd profile first, optimize second. I spent time optimizing kernels that turned out to be 5% of runtime. I'd also test correctness more rigorously from the start since debugging incorrect parallel output after running 64-process jobs is miserable. Finally, I'd document experimental methodology in real-time, not retroactively. Remembering which compiler flags I used three weeks ago is harder than it should be.


## Future Work

### SUMMA-SpMM Implementation

The next phase of this project will extend the SUMMA algorithm to **Sparse Matrix-Matrix Multiplication (SpMM)**. This computes `C = A × B` where matrices A and/or B are sparse. This operation is increasingly important in:
- Graph neural networks (GNN) inference
- Sparse deep learning model operations
- Scientific computing with sparse tensors

**Technical Challenges:**
- Adapting block distribution strategies for sparse data structures
- Minimizing communication volume when sparsity patterns are irregular
- Load balancing when block sparsity varies significantly
- Implementing efficient sparse-sparse and sparse-dense multiplication kernels

**Expected Contributions:**
- Performance comparison between dense GEMM and sparse SpMM on identical processor grids
- Analysis of how sparsity impacts communication patterns and scaling efficiency
- Implementation variants optimized for different sparsity structures (structured, random, power-law)

### Comprehensive Scaling Studies

A systematic study of strong and weak scaling characteristics across all implementations is planned to rigorously quantify parallel efficiency:

**Strong Scaling Analysis:**
- Fix problem size (matrix dimensions) and vary processor count
- Measure how execution time decreases as resources increase
- Identify scaling efficiency: `E = T₁/(N × Tₙ)` where `T₁` is sequential time, `Tₙ` is N-processor time
- Determine at what processor count communication overhead dominates, preventing further speedup
- Generate scaling curves for each matrix type (structured, random, power-law graphs)

**Weak Scaling Analysis:**
- Increase both problem size and processor count proportionally
- Measure whether per-processor work remains constant as system scales
- Identify whether the algorithm can effectively utilize additional resources given proportionally larger problems
- Test memory bandwidth scalability across nodes for SpMV operations
- Evaluate SUMMA's ability to maintain efficiency for increasingly large matrices

**Expected Outcomes:**
- Quantitative scalability limits for each implementation variant
- Optimal processor count recommendations based on problem size
- Identification of bottlenecks (computation, communication, synchronization)
- Validation of theoretical complexity models against empirical measurements
- Guidelines for selecting parallelization strategies based on available resources and problem characteristics


## Tools & Technologies

### Programming Languages & Standards
- **C (C99):** Core implementations and matrix operations
- **C++ (C++14):** CUDA kernel development
- **Bash:** Job scripting and automation

### Parallel Computing Frameworks
- **OpenMP 5.0:** Directive-based shared-memory parallelism
- **MPI (OpenMPI):** Message Passing Interface for distributed computing
- **CUDA 11.x:** NVIDIA GPU programming framework

### Development Tools
- **Compilers:** GCC (GNU Compiler Collection), NVCC (NVIDIA CUDA Compiler)
- **Build Systems:** Make, GNU Autotools
- **Debugging:** GDB, CUDA-GDB, Valgrind
- **Version Control:** Git

### HPC Infrastructure & Job Management
- **Schedulers:** Slurm Workload Manager, PBS Pro
- **Module System:** Environment Modules (Lmod) for software management
- **Parallel File Systems:** GPFS (General Parallel File System), BeeGFS, Lustre
- **Storage Management:** NFS, quota systems, automated cleanup policies
- **Network:** InfiniBand HDR/EDR (100-200 Gbps), Mellanox ConnectX-6 adapters

### Data & Visualization
- **Matrix Formats:** MatrixMarket file format, COO (Coordinate) sparse format
- **Performance Analysis:** Custom timing utilities, RDTSC for cycle-accurate measurements
- **Visualization:** Python/Matplotlib for performance graphs and scaling plots


## References

1. **Van de Geijn, R. A., & Watts, J.** (1997). SUMMA: Scalable Universal Matrix Multiplication Algorithm. *Concurrency: Practice and Experience*, 9(4), 255-274. Extended version: *PARALLEL MATRIX MULTIPLICATION: A SYSTEMATIC JOURNEY*. The University of Texas at Austin. Available at: https://www.cs.utexas.edu/~flame/pubs/SUMMA2d3dTOMS.pdf

2. **CSC548 - Parallel Systems Course** (2025). North Carolina State University, Department of Computer Science. Instructor-led exploration of parallel algorithms, distributed computing paradigms, and HPC system architectures.

3. **NCSU High Performance Computing Services**. ARC (Advanced Research Computing) and Hazel Cluster Documentation. North Carolina State University. Available at: https://hpc.ncsu.edu/main.php

4. **Davis, T. A., & Hu, Y.** (2011). The University of Florida Sparse Matrix Collection. *ACM Transactions on Mathematical Software*, 38(1), 1-25. SuiteSparse Matrix Collection: http://sparse.tamu.edu

5. **NVIDIA CUDA Programming Guide** (v11.x). NVIDIA Corporation. https://docs.nvidia.com/cuda/

6. **Gropp, W., Lusk, E., & Skjellum, A.** (2014). *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (3rd ed.). MIT Press.


## Acknowledgments

This work was conducted using the high-performance computing resources provided by **North Carolina State University High Performance Computing Services Core Facility (RRID:SCR_022168)**, including the ARC and Hazel clusters. Special thanks to the HPC support team for technical assistance and cluster access.

**Academic Context:** This project was developed as part of advanced coursework in parallel systems (CSC548) at NC State University, emphasizing practical application of parallel computing theory to real-world HPC challenges.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**Contact:** For questions about this project or collaboration opportunities, please reach out through the repository.

**Last Updated:** February 2026
