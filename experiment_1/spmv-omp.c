#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include <math.h>

// Macros for minimum and maximum
#define max(a,b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })

void usage(int argc, char **argv) {
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be a real-valued sparse matrix in MatrixMarket format.\n");
}

// Benchmark function performing SpMV with OpenMP parallel loops.
double benchmark_coo_spmv(coo_matrix *coo, float *x, float *y) {
    int num_nonzeros = coo->num_nonzeros;
    timer time_one_iteration;
    int thread_count = omp_get_max_threads();
    timer_start(&time_one_iteration);

    #pragma omp parallel for
    for (int i = 0; i < num_nonzeros; i++) {
        #pragma omp atomic
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }
    
    double estimated_time = seconds_elapsed(&time_one_iteration);
    int num_iterations = MAX_ITER;
    if (estimated_time != 0)
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
    if (num_iterations <= 0)
        num_iterations = MIN_ITER;
    
    printf("Performing %d iterations (OpenMP)\n", num_iterations);
    
    timer t;

    // printf("Total number of threads are: %d\n", thread_count);
    timer_start(&t);
    for (int iter = 0; iter < num_iterations; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < num_nonzeros; i++) {
            #pragma omp atomic
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    }

    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
    return msec_per_iteration;
}

int main(int argc, char **argv) {
    if (get_arg(argc, argv, "help") != NULL) {
        usage(argc, argv);
        return 0;
    }
    
    if (argc < 2) {
        printf("MatrixMarket file required.\n");
        return -1;
    }

    char *mm_filename = argv[1];
    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);
    // printf("MAIN Number of nonzeros is: %d \n", coo.num_nonzeros);
    
    srand(13);
    for (int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
    }
    
    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

    float *x = (float*) malloc(coo.num_cols * sizeof(float));
    float *y = (float*) calloc(coo.num_rows, sizeof(float));

    for (int i = 0; i < coo.num_cols; i++)
        x[i] = rand() / (RAND_MAX + 1.0); 
    
    double msec = benchmark_coo_spmv(&coo, x, y);
    double sec_per_iter = msec / 1000.0;
    double GFLOPs = (sec_per_iter == 0) ? 0 : (2.0 * coo.num_nonzeros / sec_per_iter) / 1e9;
    double GBYTEs = (sec_per_iter == 0) ? 0 : ((double) bytes_per_coo_spmv(&coo) / sec_per_iter) / 1e9;
    
    printf("OpenMP COO-SpMV-OMP: %.4f ms per iteration (%.2f GFLOP/s, %.1f GB/s)\n",
    msec, GFLOPs, GBYTEs);

    //printing result in a 
#ifdef TESTING
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
#endif
    
    free(x);
    free(y);
    return 0;
}
