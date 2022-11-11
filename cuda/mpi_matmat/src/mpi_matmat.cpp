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




