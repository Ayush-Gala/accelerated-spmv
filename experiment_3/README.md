# Experiment 3: SUMMA Algorithm for Dense and Sparse Matrix-Matrix Multiplication

## Overview

This experiment implements the **SUMMA (Scalable Universal Matrix Multiplication Algorithm)** for both **dense matrix-matrix multiplication (GMM)** and **sparse matrix-matrix multiplication (SPMM)** using MPI for distributed-memory parallelization.

### What is SUMMA?

SUMMA is a parallel algorithm designed for distributed matrix multiplication that achieves excellent scalability across large processor grids. Originally developed by van de Geijn and Watts, SUMMA is particularly well-suited for distributed-memory systems and forms the basis for many high-performance linear algebra libraries.

**Key Characteristics:**
- **Communication-optimal:** Minimizes data movement between processors
- **Scalable:** Works efficiently on large processor grids (√P × √P layout)
- **Memory-efficient:** Each processor stores only local submatrices
- **Flexible:** Adapts to rectangular matrices and non-square processor grids

### The SUMMA Algorithm

For computing `C = A × B` where matrices are distributed across a 2D processor grid:

**Core Principle:**
- Processors arranged in a √P × √P grid
- Matrices A, B, and C are block-distributed across the grid
- Each processor stores local blocks: `A_local`, `B_local`, `C_local`

**Algorithm Steps:**

1. **Processor Grid Setup:** Arrange P processors into a 2D grid (p_row × p_col)

2. **Block Distribution:** Partition matrices into blocks matching the processor grid
   - Process (i,j) stores: `A[i,*]`, `B[*,j]`, and computes `C[i,j]`

3. **Iterative Computation:** For each block column k:
   ```
   for k = 0 to K-1:
       // Row broadcast: Process (i,k) broadcasts A[i,k] to row i
       Broadcast A[i,k] along processor row i
       
       // Column broadcast: Process (k,j) broadcasts B[k,j] to column j
       Broadcast B[k,j] along processor column j
       
       // Local computation: Each process updates its C block
       C_local += A_local * B_local
   ```

4. **Result Assembly:** C blocks remain distributed on their owner processes

**Communication Complexity:**
- Each processor sends/receives O(n²/√P) data
- Total communication: O(n²√P) per processor
- Significantly better than naive approaches

**Computation Complexity:**
- Each processor performs O(n³/P) operations
- Perfect load balance when matrices evenly divide

---