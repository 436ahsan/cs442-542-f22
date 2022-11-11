#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#include "mpi_matmat.hpp"

// Shift A 'rank_row' columns
// Shift B 'rank_col' rows
// All pairs of A and B on a single process should be multiplied
// Then, send submatrix of A to neighboring process (rowwise)
// and submatrix of B to neighboring process (columnwise)
void mpi_matmat_cannon(double* A, double* B, double* C,
        int n, int sq_num_procs, int rank_row, int rank_col)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int size = n*n;

    double* send_A = new double[size];
    double* recv_A = new double[size];
    double* send_B = new double[size];
    double* recv_B = new double[size];
    double* tmp;

    int send_proc_A, send_proc_B;
    int recv_proc_A, recv_proc_B;
    int tag_a = 1234;
    int tag_b = 4321;
    MPI_Status status;

    for (int i = 0; i < size; i++)
        C[i] = 0;

    send_proc_A = get_proc(rank_row, rank_col-rank_row, sq_num_procs);
    send_proc_B = get_proc(rank_row-rank_col, rank_col, sq_num_procs);
    recv_proc_A = get_proc(rank_row, rank_col+rank_row, sq_num_procs);
    recv_proc_B = get_proc(rank_row+rank_col, rank_col, sq_num_procs);

    if (rank_col+rank_row >= sq_num_procs)
    {
        recv_proc_A = get_proc(rank_row, rank_col+rank_row-sq_num_procs, sq_num_procs);
        recv_proc_B = get_proc(rank_row+rank_col-sq_num_procs, rank_col, sq_num_procs);
    }

    if (rank_col - rank_row < 0)
        send_proc_A = get_proc(rank_row, rank_col-rank_row+sq_num_procs, sq_num_procs);
    if (rank_row - rank_col < 0)
        send_proc_B = get_proc(rank_row-rank_col+sq_num_procs, rank_col, sq_num_procs);


    // Initial Shift : 
    // A[rank_row, rank_row+rank_col]*B[rank_row+rank_col, rank_col]
    if (rank_row == 0)
    {
        for (int i = 0; i < size; i++)
            recv_A[i] = A[i];
    }
    else if (rank_col/rank_row % 2 == 0)
    {
        MPI_Send(A, size, MPI_DOUBLE, send_proc_A, tag_a, MPI_COMM_WORLD);
        MPI_Recv(recv_A, size, MPI_DOUBLE, recv_proc_A, tag_a, MPI_COMM_WORLD, &status);
    }
    else
    {
        MPI_Recv(recv_A, size, MPI_DOUBLE, recv_proc_A, tag_a, MPI_COMM_WORLD, &status);
        MPI_Send(A, size, MPI_DOUBLE, send_proc_A, tag_a, MPI_COMM_WORLD);
    }

    if (rank_col == 0)
    {
        for (int i = 0; i < size; i++)
            recv_B[i] = B[i];
    }
    else if (rank_row/rank_col % 2 == 0)
    {
        MPI_Send(B, size, MPI_DOUBLE, send_proc_B, tag_b, MPI_COMM_WORLD);
        MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc_B, tag_b, MPI_COMM_WORLD, &status);
    }
    else
    {
        MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc_B, tag_b, MPI_COMM_WORLD, &status);
        MPI_Send(B, size, MPI_DOUBLE, send_proc_B, tag_b, MPI_COMM_WORLD);
    }

    matmat(n, recv_A, recv_B, C);

    // Send and recv A and B from neighborhing processes in proc grid
    send_proc_A = get_proc(rank_row, rank_col+1, sq_num_procs);
    send_proc_B = get_proc(rank_row+1, rank_col, sq_num_procs);
    recv_proc_A = get_proc(rank_row, rank_col-1, sq_num_procs);
    recv_proc_B = get_proc(rank_row-1, rank_col, sq_num_procs);

    if (rank_col == sq_num_procs-1)
        send_proc_A = get_proc(rank_row, 0, sq_num_procs);
    if (rank_row == sq_num_procs-1)
        send_proc_B = get_proc(0, rank_col, sq_num_procs);
    if (rank_col == 0)
        recv_proc_A = get_proc(rank_row, sq_num_procs-1, sq_num_procs);
    if (rank_row == 0)
        recv_proc_B = get_proc(sq_num_procs-1, rank_col, sq_num_procs);



    for (int i = 1; i < sq_num_procs; i++)
    {
        tmp = recv_A;
        recv_A = send_A;
        send_A = tmp;

        tmp = recv_B;
        recv_B = send_B;
        send_B = tmp;

        if (rank_col % 2 == 0)
        {
            MPI_Send(send_A, size, MPI_DOUBLE, send_proc_A, tag_a, MPI_COMM_WORLD);
            MPI_Recv(recv_A, size, MPI_DOUBLE, recv_proc_A, tag_a, MPI_COMM_WORLD, &status);
        }
        else
        {
            MPI_Recv(recv_A, size, MPI_DOUBLE, recv_proc_A, tag_a, MPI_COMM_WORLD, &status);
            MPI_Send(send_A, size, MPI_DOUBLE, send_proc_A, tag_a, MPI_COMM_WORLD);
        }

        if (rank_row % 2 == 0)
        {
            MPI_Send(send_B, size, MPI_DOUBLE, send_proc_B, tag_b, MPI_COMM_WORLD);
            MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc_B, tag_b, MPI_COMM_WORLD, &status);
        }
        else
        {
            MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc_B, tag_b, MPI_COMM_WORLD, &status);
            MPI_Send(send_B, size, MPI_DOUBLE, send_proc_B, tag_b, MPI_COMM_WORLD);
        }

        matmat(n, recv_A, recv_B, C);
    }

    delete[] send_A;
    delete[] recv_A;
    delete[] send_B;
    delete[] recv_B;
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
