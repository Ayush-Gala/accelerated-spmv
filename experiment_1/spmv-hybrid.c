
#include <stdio.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include <mpi.h>
#include <omp.h>

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y, int world_rank)
{
    int num_nonzeros = coo->num_nonzeros;
    // printf("\tNumber of nonzeros at rank %d is : %d\n", world_rank, num_nonzeros);

    // warmup    
    timer time_one_iteration;
    timer_start(&time_one_iteration);

    #pragma omp parallel for
    for (int i = 0; i < num_nonzeros; i++){ 
        #pragma omp atomic
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }

    double estimated_time = seconds_elapsed(&time_one_iteration); 
//  printf("estimated time for once %f\n", (float) estimated_time);

    // determine # of seconds dynamically
    //global definition

    int num_iterations;
    // number of iterations will be determined by the root node and then broadcasted
    if(world_rank == 0)
    {
        num_iterations = MAX_ITER;
        if (estimated_time == 0)
            num_iterations = MAX_ITER;
        else {
            num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
        }

        printf("\tPerforming %d iterations on rank %d\n", num_iterations, world_rank);
    }

    //broadcast iterations
    MPI_Bcast(&num_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // time several SpMV iterations
    timer t;
    timer_start(&t);

    // int thread_count = omp_get_max_threads();
    //parallel region inside the for loop
    for(int j = 0; j < num_iterations; j++) {
        #pragma omp parallel for
        for (int i = 0; i < num_nonzeros; i++) {
            #pragma omp atomic
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    }
    
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    
    if(world_rank == 0)
    {
        printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)   Rank: %d\n", msec_per_iteration, GFLOPs, GBYTEs, world_rank); 
    }
    // printf("\tValue of coo.rows[2289] at rank %d is : %d\n", world_rank, y[2288]);

    return msec_per_iteration;
}

int main(int argc, char** argv)
{
    //starting MPI prog
    int world_size, world_rank;
    int global_num_nonzeros;
    char * mm_filename = NULL;

    // Initialization of environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // printf("World Size is: %d\n\n", world_rank);

    if(world_rank == 0)
    {
        if (get_arg(argc, argv, "help") != NULL) {
            usage(argc, argv);
            return 0;
        }
    
        if (argc == 1) {
            printf("Give a MatrixMarket file.\n");
            return -1;
        } else 
            mm_filename = argv[1];
    }


    coo_matrix coo;

    //for root node
    if(world_rank == 0)
    {
        read_coo_matrix(&coo, mm_filename);
        global_num_nonzeros = coo.num_nonzeros;

        // fill matrix with random values: some matrices have extreme values, 
        // which makes correctness testing difficult, especially in single precision
        srand(13);
        for(int i = 0; i < coo.num_nonzeros; i++) {
            coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
            // coo.vals[i] = 1.0;
        }

        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
        fflush(stdout);
    }

    //broadcast dimensions to all nodes
    MPI_Bcast(&coo.num_nonzeros, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //allocate x and y on every node
    float *x = (float*) malloc(coo.num_cols * sizeof(float));
    float *y = (float*) calloc(coo.num_rows, sizeof(float));

    if(world_rank == 0) {
    
        for(int i = 0; i < coo.num_cols; i++) {
            x[i] = rand() / (RAND_MAX + 1.0); 
            // x[i] = 1;
        }
    }
    
    //broadcast x to all nodes
    
    MPI_Bcast(x, coo.num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(world_rank == 0)
    {
        // Arrays to store send counts and displacements
        // Calculate chunk size for each node
        int chunk_size = coo.num_nonzeros / world_size;
        int remainder = coo.num_nonzeros % world_size;
                
        int* sendcounts = (int*)malloc(world_size * sizeof(int));
        int* displs = (int*)malloc(world_size * sizeof(int));

        // Calculate send counts and displacements
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }

        //send i-th chunk to i-th node
        // Scatter chunks of coo matrix data
        MPI_Scatterv(coo.vals, sendcounts, displs, MPI_FLOAT, 
            NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(coo.rows, sendcounts, displs, MPI_INT, 
            NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(coo.cols, sendcounts, displs, MPI_INT, 
            NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

        // modifying the local chunk for root node since it has access to all data at the moment
        int local_size = chunk_size + (world_rank < remainder ? 1 : 0);
        //global variable to remember the actual full value of num_nonzeros. Useful for debugging/testing later on
        global_num_nonzeros = coo.num_nonzeros;
        coo.num_nonzeros = local_size;

        free(sendcounts);
        free(displs);

    }
    else {
        //initialize buffer for x and chunk
        // Calculate local chunk size
        int chunk_size = coo.num_nonzeros / world_size;
        int remainder = coo.num_nonzeros % world_size;
        int local_size = chunk_size + (world_rank < remainder ? 1 : 0);

        //receive data from root
        coo.num_nonzeros = local_size;
        coo.vals = (float*)malloc(local_size * sizeof(float));
        coo.rows = (int*)malloc(local_size * sizeof(int));
        coo.cols = (int*)malloc(local_size * sizeof(int));

        // Receive scattered chunks
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, 
                     coo.vals, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, 
                     coo.rows, local_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, 
                     coo.cols, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    }

    /* Benchmarking */
    //complete y sol for each node
    // for(int i = 0; i < coo.num_rows; i++)
    //     y[i] = 0;

    double coo_gflops;
    coo_gflops = benchmark_coo_spmv(&coo, x, y, world_rank);

    // Logic to reduce all local results to root node
    float* global_y = NULL;
    if (world_rank == 0) {
        global_y = (float*)malloc(coo.num_rows * sizeof(float));
    }

    MPI_Reduce(y, global_y, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Update y pointer in root node
    if (world_rank == 0) {
        coo.num_nonzeros = global_num_nonzeros;
        free(y);
        y = global_y;
    }

#ifdef TESTING
    if(world_rank == 0)
    {
        FILE *fp;

        //first writing the output of parallel code to test_y file
        // fp = fopen("test_COO", "w");
        // printf("Writing matrix in COO format to test_COO ...");
        // fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
        // fprintf(fp, "coo.rows:\n");
        // for (int i=0; i<coo.num_nonzeros; i++)
        // {
        //   fprintf(fp, "%d  ", coo.rows[i]);
        // }

        // fprintf(fp, "\n\n");
        // fprintf(fp, "coo.cols:\n");
        // for (int i=0; i<coo.num_nonzeros; i++)
        // {
        //   fprintf(fp, "%d  ", coo.cols[i]);
        // }

        // fprintf(fp, "\n\n");
        // fprintf(fp, "coo.vals:\n");
        // for (int i=0; i<coo.num_nonzeros; i++)
        // {
        //   fprintf(fp, "%f  ", coo.vals[i]);
        // }
        // fprintf(fp, "\n");
        // fclose(fp);

        // printf("... done!\n");
        // printf("Writing x and y vectors ...");

        // fp = fopen("test_x", "w");
        // for (int i=0; i<coo.num_cols; i++)
        // {
        //   fprintf(fp, "%f\n", x[i]);
        // }
        // fclose(fp);

        fp = fopen("test_y", "w");
        for (int i=0; i<coo.num_rows; i++)
        {
          fprintf(fp, "%d\n", (int)(y[i]));
        }
        fclose(fp);

        //writing output of sequential code to seq_y
        float *seq_y = (float*) calloc(coo.num_rows, sizeof(float));
        // for(int num_iter = 0; num_iter <801; num_iter++)
        for (int i = 0; i < coo.num_nonzeros; i++){   
            seq_y[coo.rows[i]] += coo.vals[i] * x[coo.cols[i]] * 801;
        }
        fp = fopen("seq_y", "w");
        for (int i=0; i<coo.num_rows; i++)
        {
          fprintf(fp, "%d\n", (int)(seq_y[i]));
        }
        fclose(fp);
        printf("... done!\n");
    }
#endif

    //TO-DO free all buffers created in local nodes
    delete_coo_matrix(&coo);
    free(x);
    free(y);
    free(global_y);
    MPI_Finalize();
    return 0;
}
