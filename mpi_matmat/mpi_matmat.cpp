#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#include "mpi_matmat.hpp"

// Returns rank of process in 
//    (process-row 'row', process-column 'col')
int get_proc(int row, int col, int sq_procs)
{
    return row*sq_procs + col;
}

// Serial matrix-matrix multiplication
void matmat(int n, double* A, double* B, double* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i, j, k;

    double val;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            val = A[i*n+j];
            for (k = 0; k < n; k++)
            {
                C[i*n+k] += val * B[j*n+k];
            }
        }
    }
 
}

// Calculates sum of matrix for error checking
double mat_sum(int n, double* C)
{
    double sum = 0;
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            sum += C[i*n+j];
        }
    }
    return sum;
}


// Main Method : 
//     Splits processes into a process grid
//         - rank_row : row of process in process grid
//         - rank_col : column of process in process grid
//     Creates three local matrices, A, B, and C
//     Times all three implementations of parallel DGEMM
//     Prints timings of methods
int main(int argc, char* argv[])
{

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get rank of process and number of processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Make sure matrix dimension is in argv
    if (argc <= 1)
    {   
        if (rank == 0)
            printf("Pass Matrix Dimension as Command Line Argument!\n");
        MPI_Finalize();
        return 1;
    }

    // Grab global dimension of matrices (A, B, C)
    int N = atoi(argv[1]);

    // Calculate how many process rows/cols in process-grid
    int sq_num_procs = sqrt(num_procs);
    if (sq_num_procs*sq_num_procs != num_procs)
    {
        if (rank == 0) 
            printf("Number of processes needs to be a square\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate variables
    // - rank_row : process row
    // - rank_col : process col
    // - n : local (per-process) matrix dimension
    int rank_row = rank / sq_num_procs;
    int rank_col = rank % sq_num_procs;
    int n = N / sq_num_procs;
    int size = n*n;

    if (n*n*num_procs != N*N)
    {
        if (rank == 0) 
            printf("Cannot evenly split %d rows and cols over %d processes\n",
                    N, num_procs);
        MPI_Finalize();
        return 1;
    }

    // Allocate three local matrices (A, B, C)
    double* A = new double[size];
    double* B = new double[size];
    double* C = new double[size];

    // Initialize matrices A and B 
    int first_i = rank_row*N;
    int first_j = rank_col;
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
            B[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
        }
    }
    
    double sum_C, total_sum_C;
    double start, end;

    // Time Simple Method
    mpi_matmat_simple(A, B, C, n, sq_num_procs, rank_row, rank_col);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    mpi_matmat_simple(A, B, C, n, sq_num_procs, rank_row, rank_col);
    end = MPI_Wtime() - start;
    sum_C = mat_sum(n, C);
    MPI_Reduce(&sum_C, &total_sum_C, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simple Method : sumC %e, Elapsed Time %e\n", total_sum_C, start);

    // Time Fox's Method
    mpi_matmat_fox(A, B, C, n, sq_num_procs, rank_row, rank_col);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    mpi_matmat_fox(A, B, C, n, sq_num_procs, rank_row, rank_col);
    end = MPI_Wtime() - start;
    sum_C = mat_sum(n, C);
    MPI_Reduce(&sum_C, &total_sum_C, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Fox's Method : sumC %e, Elapsed Time %e\n", total_sum_C, start);

    // Time Cannon's Method
    mpi_matmat_cannon(A, B, C, n, sq_num_procs, rank_row, rank_col);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    mpi_matmat_cannon(A, B, C, n, sq_num_procs, rank_row, rank_col);
    end = MPI_Wtime() - start;
    sum_C = mat_sum(n, C);
    MPI_Reduce(&sum_C, &total_sum_C, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Cannon's Method : sumC %e, Elapsed Time %e\n", total_sum_C, start);
     

    delete[] A;
    delete[] B;
    delete[] C;

    MPI_Finalize();
    return 0;
}
