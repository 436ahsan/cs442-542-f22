#include "mpi_matmat.hpp"

// Simplest multiply of two parallal matrices with 2D partition
// Send all values of A to processes holding other parts of the same rows of A
// Send all values of B to processes holding other parts of the same rows of B
// Recv matching pairs of A and B, and multiply these together
void mpi_matmat_simple(double* A, double* B, double* C,
        int n, int sq_num_procs, int rank_row, int rank_col)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int first_rank_row = rank_row*sq_num_procs;

    int proc;
    int tag_a = 1234;
    int tag_b = 4321;
    int size = n*n;

    // Initilize C to 0
    for (int i = 0; i < size; i++)
        C[i] = 0;

    double* a_send_buffer = new double[sq_num_procs*size];
    double* b_send_buffer = new double[sq_num_procs*size];
    double* recv_A = new double[size];
    double* recv_B = new double[size];
    MPI_Request* a_requests = new MPI_Request[sq_num_procs];
    MPI_Request* b_requests = new MPI_Request[sq_num_procs];


    // Send local portion of A to every process in rank_row
    // Copied to a_send_buffer because MPI_Isend is not blocking
    // Cannot reuse a_send_buffer until after MPI_Waitall
    for (int i = 0; i < sq_num_procs; i++)
    {
        proc = get_proc(rank_row, i, sq_num_procs);
        for (int j = 0; j < size; j++)
            a_send_buffer[i*size+j] = A[j];
        MPI_Isend(&(a_send_buffer[i*size]), size, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &(a_requests[i]));
    }

    // Send local portion of B to every process in rank_col
    // Copied to b_send_buffer because MPI_Isend is not blocking
    // Cannot reuse b_send_buffer until after MPI_Waitall
    for (int i = 0; i < sq_num_procs; i++)
    {
        proc = get_proc(i, rank_col, sq_num_procs);
        for (int j = 0; j < size; j++)
            b_send_buffer[i*size+j] = B[j];
        MPI_Isend(&(b_send_buffer[i*size]), size, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &(b_requests[i]));
    }


    // Multiply C +=  A[rank_row, i] * B[i, rank_col]
    for (int i = 0; i < sq_num_procs; i++)
    {

        // Recv A from proc in (rank_row, i) 
        proc = get_proc(rank_row, i, sq_num_procs);
        MPI_Recv(recv_A, size, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Recv B from proc in (i, rank_col)
        proc = get_proc(i, rank_col, sq_num_procs);
        MPI_Recv(recv_B, size, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Locally multiply C += recv_A * recv_B
        matmat(n, recv_A, recv_B, C);
    }

    // Wait for all of my sends to complete
    // Must wait before freeing a_send_buffer, b_send_buffer
    MPI_Waitall(sq_num_procs, a_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(sq_num_procs, b_requests, MPI_STATUSES_IGNORE);

    delete[] recv_A;
    delete[] recv_B;
    delete[] a_send_buffer;
    delete[] b_send_buffer;
    delete[] a_requests;
    delete[] b_requests;
}

