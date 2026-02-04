#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void distribute_matrix_blocks(float *global_matrix, float *local_block, int m, int n, int local_m, int local_n, MPI_Comm comm_2d, int grid_size, int rank, int transpose) {
  // Create datatype for the global matrix blocks
  MPI_Datatype block_type, resized_block_type;
  int starts[2] = {0, 0};
  int subsizes[2] = {local_m, local_n};
  int bigsizes[2] = {m, n};
  
  //creating data type for blocks
  MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &block_type);
  MPI_Type_commit(&block_type);
  
  // Resize the datatype to account for the block's actual size
  MPI_Type_create_resized(block_type, 0, sizeof(float), &resized_block_type);
  MPI_Type_commit(&resized_block_type);
  
  // Calculate send counts and displacements
  int *send_counts = (int *)malloc(grid_size * grid_size * sizeof(int));
  int *displs = (int *)malloc(grid_size * grid_size * sizeof(int));

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
  
  // Scatter the blocks to all processes
  MPI_Scatterv(global_matrix, send_counts, displs, resized_block_type, local_block, local_m * local_n, MPI_FLOAT, 0, comm_2d);

  /*
  --------------------- TESTING BLOCK DISTRIBUTION -------------------
  */

  // //printing the global matrix
  // if(rank == 0){
  //   for(int i=0; i<m*n; i++){
  //     if(i%m == 0){
  //       printf("\n");
  //     }
  //     printf("%.2f ",global_matrix[i]);
  //   }
  //   printf("\n");
  //   printf("\n");
  //   printf("m: %d | n: %d | local_m: %d | local_n: %d\n\n", m, n, local_m, local_n);
  // }

  // MPI_Barrier(comm_2d);

  // // Get the 2D coordinates of this process
  // int coords[2];
  // MPI_Cart_coords(comm_2d, rank, 2, coords);
  
  // // Barrier to synchronize output
  // MPI_Barrier(comm_2d);
  
  // // Each process prints its received block
  // for (int proc_rank = 0; proc_rank < grid_size * grid_size; proc_rank++) {
  //   if (rank == proc_rank) {
  //       printf("Process %d (coords: %d,%d) received block:\n", rank, coords[0], coords[1]);
  //       for (int i = 0; i < local_m; i++) {
  //           printf("  ");
  //           for (int j = 0; j < local_n; j++) {
  //               printf("%.2f ", local_block[i * local_n + j]);
  //           }
  //           printf("\n");
  //       }
  //       printf("\n");
  //       fflush(stdout);
  //   }
  //   // Barrier to ensure ordered output
  //   MPI_Barrier(comm_2d);
  // }

   /*
  --------------------- TESTING BLOCK DISTRIBUTION -------------------
  */
  
  // Clean up
  MPI_Type_free(&block_type);
  MPI_Type_free(&resized_block_type);
  free(send_counts);
  free(displs);
}


/*
----------------------------------------------------------------------------
------------------------- STATIONARY-C -------------------------------------
----------------------------------------------------------------------------
*/

void summa_stationary_c(int m, int n, int k, int nprocs, int rank) {

  int grid_size = (int)sqrt(nprocs);
  
  // Create 2D process grid
  int dims[2] = {grid_size, grid_size};
  int periods[2] = {0, 0};
  int coords[2];
  MPI_Comm grid_comm, row_comm, col_comm;
  
  //creatimg cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
  
  //getting process coordinates
  MPI_Cart_coords(grid_comm, rank, 2, coords);
  
  //recording the row and column of the process in the grid
  int proc_row = coords[0];
  int proc_col = coords[1];

  // Create row communicators (all processes in the same row)
  MPI_Comm_split(grid_comm, proc_row, proc_col, &row_comm);
    
  // Create column communicators (all processes in the same column)
  MPI_Comm_split(grid_comm, proc_col, proc_row, &col_comm);

   /*
  --------------------- TESTING GRID COMMUNICATORS -------------------
  */
  
  // // Verify the grid and communicators
  // int row_rank, row_size, col_rank, col_size;
  // MPI_Comm_rank(row_comm, &row_rank);
  // MPI_Comm_size(row_comm, &row_size);
  // MPI_Comm_rank(col_comm, &col_rank);
  // MPI_Comm_size(col_comm, &col_size);
  
  // printf("Process %d: Grid position (%d,%d), Row rank %d/%d, Col rank %d/%d\n",
  //        rank, my_row, my_col, row_rank, row_size, col_rank, col_size);

   /*
  --------------------- TESTING GRID COMMUNICATORS -------------------
  */

  // Calculate local block sizes
  int local_m = m / grid_size;
  int local_n = n / grid_size;
  int local_k = k / grid_size;

  //local matrice buffer
  float *A_local = (float *)malloc(local_m * local_k * sizeof(float));
  float *B_local = (float *)malloc(local_k * local_n * sizeof(float));
  float *C_local = (float *)calloc(local_m * local_n, sizeof(float));

  // temp buffers for broadcasting
  float *A_temp = (float *)malloc(local_m * local_k * sizeof(float));
  float *B_temp = (float *)malloc(local_k * local_n * sizeof(float));

  //creating matrices on root prcoess
  float *A = NULL;
  float *B = NULL;
  float *C = NULL;

  if(rank == 0) {
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
    C = (float *)calloc(m * n, sizeof(float));
  }

  double parallel_start_time, parallel_end_time, blk_dist_end_time, main_comp_end_time, serial_start_time ,serial_end_time;
  parallel_start_time = MPI_Wtime();

  // Distribute matrix blocks
  distribute_matrix_blocks(A, A_local, m, k, local_m, local_k, grid_comm, grid_size, rank, 0);
  distribute_matrix_blocks(B, B_local, k, n, local_k, local_n, grid_comm, grid_size, rank, 0);
  
  blk_dist_end_time = MPI_Wtime();

  double broadcast_start, broadcast_end, broadcast_sum;
  broadcast_sum = (double)0;

  // Main computation loop
  for (int s = 0; s < grid_size; s++) {
    // Determine the process in current row that holds the required A block
    int A_owner_col = s;
    
    // Determine the process in current column that holds the required B block
    int B_owner_row = s;
    
    // Copy local A block to A_temp if this process owns it
    if (proc_col == A_owner_col) {
        memcpy(A_temp, A_local, local_m * local_k * sizeof(float));
    }
    
    // Copy local B block to B_temp if this process owns it
    if (proc_row == B_owner_row) {
        memcpy(B_temp, B_local, local_k * local_n * sizeof(float));
    }
    
    broadcast_start = MPI_Wtime();

    // Broadcast A_temp along the row
    MPI_Bcast(A_temp, local_m * local_k, MPI_FLOAT, A_owner_col, row_comm);
    
    // Broadcast B_temp along the column
    MPI_Bcast(B_temp, local_k * local_n, MPI_FLOAT, B_owner_row, col_comm);

    broadcast_end = MPI_Wtime();
    broadcast_sum += (broadcast_end - broadcast_start);
    
    // Perform local matrix multiplication: C_local += A_temp * B_temp
    for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < local_n; j++) {
            for (int p = 0; p < local_k; p++) {
                C_local[i * local_n + j] += A_temp[i * local_k + p] * B_temp[p * local_n + j];
            }
        }
    }
    
    // Synchronize before next iteration
    MPI_Barrier(grid_comm);
  }

  main_comp_end_time = MPI_Wtime();

   /*
  --------------------- TESTING C_local values -------------------
  */
  
  // // Print local C matrix from each process
  // // Use barriers to ensure ordered output
  // MPI_Barrier(grid_comm);
  // for (int p = 0; p < nprocs; p++) {
  //   if (rank == p) {
  //     printf("Process %d (coords: %d,%d) computed C_local:\n", rank, proc_row, proc_col);
  //     for (int i = 0; i < local_m; i++) {
  //         printf("  ");
  //         for (int j = 0; j < local_n; j++) {
  //             printf("%.2f ", C_local[i * local_n + j]);
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //     fflush(stdout);
  //   }
  //   MPI_Barrier(grid_comm);
  // }
  // MPI_Barrier(grid_comm);

   /*
  --------------------- TESTING C_local values -------------------
  */

  // Gather results back to root
  MPI_Datatype block_type, resized_block_type;
  int starts[2] = {0, 0};
  int subsizes[2] = {local_m, local_n};
  int bigsizes[2] = {m, n};

  MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &block_type);
  MPI_Type_commit(&block_type);

  MPI_Type_create_resized(block_type, 0, sizeof(float), &resized_block_type);
  MPI_Type_commit(&resized_block_type);

  int *recv_counts = NULL;
  int *displs = NULL;

  if (rank == 0) {
      recv_counts = (int *)malloc(grid_size * grid_size * sizeof(int));
      displs = (int *)malloc(grid_size * grid_size * sizeof(int));

      for (int i = 0; i < grid_size; i++) {
          for (int j = 0; j < grid_size; j++) {
              recv_counts[i * grid_size + j] = 1;
              displs[i * grid_size + j] = i * local_m * n + j * local_n;
          }
      }
  }

  MPI_Gatherv(C_local, local_m * local_n, MPI_FLOAT,
              C, recv_counts, displs, resized_block_type,
              0, grid_comm);

  //end of complete parallel execution
  parallel_end_time = MPI_Wtime();

  if (rank == 0) {
    printf("\n Initial block distribution time: %.4f seconds \n", blk_dist_end_time - parallel_start_time);
    printf("\n Execution time of main computation: %.4f seconds \n", main_comp_end_time - blk_dist_end_time);
    printf("\n Total time spent in messaging in main computation: %.4f seconds \n", broadcast_sum);
    printf("\n MPI_Gatherv execution time: %.4f seconds \n", parallel_end_time - main_comp_end_time);
    printf("\n Parallel Compute complete in: %.4f seconds \n\n ", parallel_end_time - parallel_start_time);
    
    serial_start_time = MPI_Wtime();
    verify_result(C, A, B, m, n, k);
    serial_end_time = MPI_Wtime();

    printf("\n Serial matrix multiplication time is: %.4f seconds\n", serial_end_time - serial_start_time);
    
  }

  // Clean up
  free(A_local);
  free(B_local);
  free(C_local);
  free(A_temp);
  free(B_temp);

  if (rank == 0) {
      free(A);
      free(B);
      free(C);
      free(recv_counts);
      free(displs);
  }

  MPI_Type_free(&block_type);
  MPI_Type_free(&resized_block_type);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&grid_comm);
}

/*
----------------------------------------------------------------------------
------------------------- STATIONARY-A -------------------------------------
----------------------------------------------------------------------------
*/

void summa_stationary_a(int m, int n, int k, int nprocs, int rank) {
  // TODO: Implement SUMMA algorithm with stationary A

  int grid_size = (int)sqrt(nprocs);
  
  // Create 2D process grid
  int dims[2] = {grid_size, grid_size};
  int periods[2] = {0, 0};
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

  // Calculate local block sizes
  int local_m = m / grid_size;
  int local_n = n / grid_size;
  int local_k = k / grid_size;

  // Local matrices buffers
  float *A_local = (float *)malloc(local_m * local_k * sizeof(float));
  float *B_local = (float *)malloc(local_k * local_n * sizeof(float));
  float *C_local = (float *)calloc(local_m * local_n, sizeof(float));

  // Temp buffers for broadcasting and computation
  float *B_temp = (float *)malloc(local_k * local_n * sizeof(float));
  float *C_temp = (float *)calloc(local_m * local_n, sizeof(float));

  // Creating matrices on root process
  float *A = NULL;
  float *B = NULL;
  float *C = NULL;

  if (rank == 0) {
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
    C = (float *)calloc(m * n, sizeof(float));
  }

  double parallel_start_time, parallel_end_time, blk_dist_end_time, main_comp_end_time, serial_start_time, serial_end_time;
  parallel_start_time = MPI_Wtime();

  // Distribute matrix blocks - A in the normal way, B in the transposed way
  distribute_matrix_blocks(A, A_local, m, k, local_m, local_k, grid_comm, grid_size, rank, 0);
  distribute_matrix_blocks(B, B_local, k, n, local_k, local_n, grid_comm, grid_size, rank, 1);
  
  blk_dist_end_time = MPI_Wtime();

  double broadcast_start, broadcast_end, broadcast_sum;
  broadcast_sum = (double)0;

  // Main computation loop for stationary-A
  for (int s = 0; s < grid_size; s++) {
    // Determine the process in current row that holds the required A block
    int A_owner_col = s;
    
    // Determine the process in current column that holds the required B block (transposed distribution)
    int B_owner_row = s;
    
    // Copy local B block to B_temp if this process owns it
    if (proc_row == B_owner_row) {
        memcpy(B_temp, B_local, local_k * local_n * sizeof(float));
    }
    
    broadcast_start = MPI_Wtime();
    // Broadcast B_temp along the column
    MPI_Bcast(B_temp, local_k * local_n, MPI_FLOAT, B_owner_row, col_comm);

    broadcast_end = MPI_Wtime();
    broadcast_sum += (broadcast_end - broadcast_start);
    
    // Reset C_temp for this iteration
    memset(C_temp, 0, local_m * local_n * sizeof(float));
    
    // Perform local matrix multiplication: C_temp = A_local * B_temp
    for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < local_n; j++) {
            for (int p = 0; p < local_k; p++) {
                C_temp[i * local_n + j] += A_local[i * local_k + p] * B_temp[p * local_n + j];
            }
        }
    }
    
    // Use MPI_Reduce_scatter to accumulate C_temp values to C_local at the A_owner_col
    // First, create array of recvcounts for each process in the row
    int *recvcounts = (int *)malloc(grid_size * sizeof(int));
    for (int i = 0; i < grid_size; i++) {
        recvcounts[i] = (i == A_owner_col) ? local_m * local_n : 0;
    }
    
    broadcast_start = MPI_Wtime();

    // Reduce_scatter the C_temp values along the row to the A_owner_col process
    MPI_Reduce_scatter(C_temp, C_local, recvcounts, MPI_FLOAT, MPI_SUM, row_comm);
    
    broadcast_end = MPI_Wtime();
    broadcast_sum += (broadcast_end - broadcast_start);

    free(recvcounts);
    // Synchronize before next iteration
    MPI_Barrier(grid_comm);
  }

  main_comp_end_time = MPI_Wtime();

  // Gather results back to root
  MPI_Datatype block_type, resized_block_type;
  int starts[2] = {0, 0};
  int subsizes[2] = {local_m, local_n};
  int bigsizes[2] = {m, n};

  MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &block_type);
  MPI_Type_commit(&block_type);

  MPI_Type_create_resized(block_type, 0, sizeof(float), &resized_block_type);
  MPI_Type_commit(&resized_block_type);

  int *recv_counts = NULL;
  int *displs = NULL;

  if (rank == 0) {
      recv_counts = (int *)malloc(grid_size * grid_size * sizeof(int));
      displs = (int *)malloc(grid_size * grid_size * sizeof(int));

      for (int i = 0; i < grid_size; i++) {
          for (int j = 0; j < grid_size; j++) {
              recv_counts[i * grid_size + j] = 1;
              displs[i * grid_size + j] = i * local_m * n + j * local_n;
          }
      }
  }

  MPI_Gatherv(C_local, local_m * local_n, MPI_FLOAT,
              C, recv_counts, displs, resized_block_type,
              0, grid_comm);

  parallel_end_time = MPI_Wtime();

  if (rank == 0) {
    printf("\n Initial block distribution time: %.4f seconds \n", blk_dist_end_time - parallel_start_time);
    printf("\n Execution time of main computation: %.4f seconds \n", main_comp_end_time - blk_dist_end_time);
    printf("\n Total time spent in messaging in main computation: %.4f seconds \n", broadcast_sum);
    printf("\n MPI_Gatherv execution time: %.4f seconds \n", parallel_end_time - main_comp_end_time);
    printf("\n Parallel Compute complete in: %.4f seconds\n ", parallel_end_time - parallel_start_time);

    serial_start_time = MPI_Wtime();
    verify_result(C, A, B, m, n, k);
    serial_end_time = MPI_Wtime();

    printf("\n Serial matrix multiplication time is: %.4f seconds\n", serial_end_time - serial_start_time);

  }

  // Clean up
  free(A_local);
  free(B_local);
  free(C_local);
  free(B_temp);
  free(C_temp);

  if (rank == 0) {
      free(A);
      free(B);
      free(C);
      free(recv_counts);
      free(displs);
  }

  MPI_Type_free(&block_type);
  MPI_Type_free(&resized_block_type);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&grid_comm);
}

/*
----------------------------------------------------------------------------
------------------------- STATIONARY-B -------------------------------------
----------------------------------------------------------------------------
*/

void summa_stationary_b(int m, int n, int k, int nprocs, int rank) {
  
  int grid_size = (int)sqrt(nprocs);
  
  // Create 2D process grid
  int dims[2] = {grid_size, grid_size};
  int periods[2] = {0, 0};
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

  // Calculate local block sizes
  int local_m = m / grid_size;
  int local_n = n / grid_size;
  int local_k = k / grid_size;

  // Local matrices buffers
  float *A_local = (float *)malloc(local_m * local_k * sizeof(float));
  float *B_local = (float *)malloc(local_k * local_n * sizeof(float));
  float *C_local = (float *)calloc(local_m * local_n, sizeof(float));

  // Temp buffers for broadcasting and computation
  float *A_temp = (float *)malloc(local_m * local_k * sizeof(float));
  float *C_temp = (float *)calloc(local_m * local_n, sizeof(float));

  // Creating matrices on root process
  float *A = NULL;
  float *B = NULL;
  float *C = NULL;

  if (rank == 0) {
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
    C = (float *)calloc(m * n, sizeof(float));
  }

  double parallel_start_time, parallel_end_time, blk_dist_end_time, main_comp_end_time, serial_start_time, serial_end_time;
  parallel_start_time = MPI_Wtime();

  // Distribute matrix blocks - A in the normal way, B in the transposed way
  distribute_matrix_blocks(A, A_local, m, k, local_m, local_k, grid_comm, grid_size, rank, 1);
  distribute_matrix_blocks(B, B_local, k, n, local_k, local_n, grid_comm, grid_size, rank, 0);
  
  blk_dist_end_time = MPI_Wtime();

  double broadcast_start, broadcast_end, broadcast_sum;
  broadcast_sum = (double)0;

  // Main computation loop for stationary-A
  for (int s = 0; s < grid_size; s++) {
    // Determine the process in current row that holds the required A block
    int A_owner_col = s;
    
    // Determine the process in current column that holds the required B block (transposed distribution)
    int B_owner_row = s;
    
    // Copy local A block to A_temp if this process owns it
    if (proc_col == A_owner_col) {
      memcpy(A_temp, A_local, local_m * local_k * sizeof(float));
    }
    
    broadcast_start = MPI_Wtime();

    // Broadcast A_temp along the row
    MPI_Bcast(A_temp, local_m * local_k, MPI_FLOAT, A_owner_col, row_comm);

    broadcast_end = MPI_Wtime();
    broadcast_sum += (broadcast_end - broadcast_start);
    
    // Reset C_temp for this iteration
    memset(C_temp, 0, local_m * local_n * sizeof(float));
    
    // Perform local matrix multiplication: C_temp = A_local * B_temp
    for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < local_n; j++) {
            for (int p = 0; p < local_k; p++) {
                C_temp[i * local_n + j] += A_temp[i * local_k + p] * B_local[p * local_n + j];
            }
        }
    }
    
    // Use MPI_Reduce_scatter to accumulate C_temp values to C_local at the B_owner_row
    // First, create array of recvcounts for each process in the row
    int *recvcounts = (int *)malloc(grid_size * sizeof(int));
    for (int i = 0; i < grid_size; i++) {
        recvcounts[i] = (i == B_owner_row) ? local_m * local_n : 0;
    }
    
    broadcast_start = MPI_Wtime();

    // Reduce_scatter the C_temp values along the row to the B_owner_row process
    MPI_Reduce_scatter(C_temp, C_local, recvcounts, MPI_FLOAT, MPI_SUM, col_comm);
    
    broadcast_end = MPI_Wtime();
    broadcast_sum += (broadcast_end - broadcast_start);

    free(recvcounts);
    // Synchronize before next iteration
    MPI_Barrier(grid_comm);
  }

  main_comp_end_time = MPI_Wtime();

  // Gather results back to root
  MPI_Datatype block_type, resized_block_type;
  int starts[2] = {0, 0};
  int subsizes[2] = {local_m, local_n};
  int bigsizes[2] = {m, n};

  MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &block_type);
  MPI_Type_commit(&block_type);

  MPI_Type_create_resized(block_type, 0, sizeof(float), &resized_block_type);
  MPI_Type_commit(&resized_block_type);

  int *recv_counts = NULL;
  int *displs = NULL;

  if (rank == 0) {
      recv_counts = (int *)malloc(grid_size * grid_size * sizeof(int));
      displs = (int *)malloc(grid_size * grid_size * sizeof(int));

      for (int i = 0; i < grid_size; i++) {
          for (int j = 0; j < grid_size; j++) {
              recv_counts[i * grid_size + j] = 1;
              displs[i * grid_size + j] = i * local_m * n + j * local_n;
          }
      }
  }

  MPI_Gatherv(C_local, local_m * local_n, MPI_FLOAT,
              C, recv_counts, displs, resized_block_type,
              0, grid_comm);

  parallel_end_time = MPI_Wtime();

  if (rank == 0) {
    printf("\n Initial block distribution time: %.4f seconds \n", blk_dist_end_time - parallel_start_time);
    printf("\n Execution time of main computation: %.4f seconds \n", main_comp_end_time - blk_dist_end_time);
    printf("\n Total time spent in messaging in main computation: %.4f seconds \n", broadcast_sum);
    printf("\n MPI_Gatherv execution time: %.4f seconds \n", parallel_end_time - main_comp_end_time);
    printf("\n Parallel Compute complete in: %.4f seconds\n ", parallel_end_time - parallel_start_time);

    serial_start_time = MPI_Wtime();
    verify_result(C, A, B, m, n, k);
    serial_end_time = MPI_Wtime();

    printf("\n Serial matrix multiplication time is: %.4f seconds\n", serial_end_time - serial_start_time);

  }

  // Clean up
  free(A_local);
  free(B_local);
  free(C_local);
  free(A_temp);
  free(C_temp);

  if (rank == 0) {
      free(A);
      free(B);
      free(C);
      free(recv_counts);
      free(displs);
  }

  MPI_Type_free(&block_type);
  MPI_Type_free(&resized_block_type);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&grid_comm);
}

int main(int argc, char *argv[]) {
  // Initialize the MPI environment
  // TODO: Initialize MPI
  MPI_Init(&argc, &argv);

  // Get the rank of the process
  // TODO: Get the rank of the current process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the number of processes
  // TODO: Get the total number of processes
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if(rank == 0){
    printf("Total number of processors are: %d", nprocs);
  }

  SummaOpts opts;
  if(rank == 0) {
    opts = parse_args(argc, argv);
  }

  // Broadcast options to all processes
  // TODO: Broadcast the parsed options to all processes
  MPI_Bcast(&opts, sizeof(SummaOpts), MPI_BYTE, 0, MPI_COMM_WORLD);

  // Check if number of processes is a perfect square
  // TODO: Check if the number of processes forms a perfect square grid
  int grid_size = (int)sqrt(nprocs);
  if (grid_size * grid_size != nprocs) {
    if (rank == 0) {
      printf("Error: Number of processes (%d) must be a perfect square\n", nprocs);
    }
    MPI_Finalize();
    return 1;
  }
  
  // Check if matrix dimensions are compatible with grid size
  if (opts.m % grid_size != 0 || opts.n % grid_size != 0 ||
      opts.k % grid_size != 0) {
    printf("Error: Matrix dimensions must be divisible by grid size (%d)\n",
           grid_size);
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    printf("\nMatrix Dimensions:\n");
    printf("A: %d x %d\n", opts.m, opts.k);
    printf("B: %d x %d\n", opts.k, opts.n);
    printf("C: %d x %d\n", opts.m, opts.n);
    printf("Grid size: %d x %d\n", grid_size, grid_size);
    printf("Block size: %d\n", opts.block_size);
    printf("Algorithm: Stationary %c\n", opts.stationary);
    printf("Verbose: %s\n", opts.verbose ? "true" : "false");
  }

  double program_start_time, program_end_time;
  program_start_time = MPI_Wtime();
  // Call the appropriate SUMMA function based on algorithm variant
  if (opts.stationary == 'A' || opts.stationary == 'a') {
    summa_stationary_a(opts.m, opts.n, opts.k, nprocs, rank);
  }
  else if (opts.stationary == 'B' || opts.stationary == 'b') {
    summa_stationary_b(opts.m, opts.n, opts.k, nprocs, rank);
  }
  else if (opts.stationary == 'C' || opts.stationary == 'c') {
    summa_stationary_c(opts.m, opts.n, opts.k, nprocs, rank);
  }
  else {
    if (rank == 0) {
      printf("Error: Unknown stationary option '%c'. Use 'A', 'B', or 'C'.\n", opts.stationary);
    }
    MPI_Finalize();
    return 1;
  }

  program_end_time = MPI_Wtime();
  if(rank == 0) {
    printf("\n Time taken for complete program execution is: %.4f seconds \n", program_end_time - program_start_time);
  }

  MPI_Finalize();
  return 0;
}