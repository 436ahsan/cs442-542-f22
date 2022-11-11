#include "mpi_matmat.hpp"

// Fox's Algorithm 
void mpi_matmat_fox(double* A, double* B, double* C,
        int n, int sq_num_procs, int rank_row, int rank_col)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int proc;
    int tag_a = 1234;
    int tag_b = 4321;
    int size = n*n;
    MPI_Status status;

    double* recv_A = new double[size];
    double* send_B = new double[size];
    double* recv_B = new double[size];
    double* tmp;

    // Initialize C as 0
    for (int i = 0; i < size; i++)
        C[i] = 0;

    // Send local A to all processes in row 
    double* a_send_buffer = new double[sq_num_procs*size];
    MPI_Request* a_requests = new MPI_Request[sq_num_procs];
    for (int i = 0; i < sq_num_procs; i++)
    {
        proc = get_proc(rank_row, i, sq_num_procs);
        for (int j = 0; j < size; j++)
            a_send_buffer[i*size+j] = A[j];
        MPI_Isend(&(a_send_buffer[i*size]), size, MPI_DOUBLE, proc, tag_a, 
                MPI_COMM_WORLD, &(a_requests[i]));
    }


    // Find processes for communicating B:
    // - send_proc : process to which I send local portion of B
    // - recv_proc : process from which I recv new local portion of B
    // Check to make sure I wrap around (proc >= 0, < sq_num_procs) 
    int send_proc = get_proc(rank_row+1, rank_col, sq_num_procs);
    int recv_proc = get_proc(rank_row-1, rank_col, sq_num_procs);
    if (rank_row == sq_num_procs - 1)
    {
        send_proc = get_proc(0, rank_col, sq_num_procs);
    }
    if (rank_row == 0)
    {
        recv_proc = get_proc(sq_num_procs-1, rank_col, sq_num_procs);
    }

    // Always use initial local portion of B first
    // Copy this into recv_B to use
    for (int i = 0; i < size; i++)
        recv_B[i] = B[i];
   
    // Receive portion of A corresponding to local portion of B
    // - place in 'recv_A'
    // - A[rank_row, rank_row]
    int pos = rank_row;
    proc = get_proc(rank_row, pos, sq_num_procs);
    MPI_Recv(recv_A, size, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Multiple recv_A * recv_B
    matmat(n, recv_A, recv_B, C);



    // For each iteration, receive new portions of A and B and multiply together
    for (int i = 1; i < sq_num_procs; i++)
    {
        // Will send what is currently in recv_B
        // Will recv new portion of B
        // Just swap these pointers (cheap)
        tmp = send_B;
        send_B = recv_B;
        recv_B = tmp;

        // Get portion of A one more position to left
        pos = pos - 1;
        if (pos < 0) 
            pos += sq_num_procs;
        proc = get_proc(rank_row, pos, sq_num_procs);
        MPI_Recv(recv_A, size, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &status);

        // Send portion of B to send_proc
        // Recv portion of B from recv_proc
        if (rank_row % 2 == 0)
        {
            MPI_Send(send_B, size, MPI_DOUBLE, send_proc, tag_b, MPI_COMM_WORLD);
            MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc, tag_b, MPI_COMM_WORLD, &status);
        }
        else
        {
            MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc, tag_b, MPI_COMM_WORLD, &status);
            MPI_Send(send_B, size, MPI_DOUBLE, send_proc, tag_b, MPI_COMM_WORLD);
        }

        // Multiply C += recv_A * recv_B
        matmat(n, recv_A, recv_B, C);
    }

    // Wait for all of the sends of A to complete
    MPI_Waitall(sq_num_procs, a_requests, MPI_STATUSES_IGNORE);

    delete[] recv_A;
    delete[] send_B;
    delete[] recv_B;
    
    delete[] a_send_buffer;
    delete[] a_requests;

}

