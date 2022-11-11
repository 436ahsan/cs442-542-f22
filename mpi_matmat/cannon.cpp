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
    int send_proc_A, send_proc_B;
    int recv_proc_A, recv_proc_B;
    int tag_a = 1234;
    int tag_b = 4321;
    MPI_Status status;

    double* send_A = new double[size];
    double* recv_A = new double[size];
    double* send_B = new double[size];
    double* recv_B = new double[size];
    double* tmp;

    for (int i = 0; i < size; i++)
        C[i] = 0;


    // Determine Send and Recv Processes for Initial Shift
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
        

    // 1. Perform Initial Shift : 
    // Goal : A[rank_row, rank_row+rank_col]*B[rank_row+rank_col, rank_col]
    

    // 2. Perform local matrix-multiplication
    // on submatrices received in initial shift


    // 3. Determine new values for send_proc_A/B, recv_proc_A/B
    // Send A to [rank_row, rank_col+1]
    // Send B to [rank_row+1, rank_col]
    // Recv A from [rank_row, rank_col-1]
    // Recv B from [rank_row-1, rank_col]
    // Make sure to check bounds (wrap around if >= sq_num_procs or < 0)


    // 4. For each iteration, send and recv A, B, and perform multiplication
    for (int i = 1; i < sq_num_procs; i++)
    {    
        // 4a. Send A to send_proc_A
        // 4b. Recv new A from recv_proc_A

        // 4c. Send B to send_proc_B
        // 4c. Recv new B from recv_proc_B


        // 4e. Local matrix multiplication C += recv_A * recv_B
    }

    delete[] send_A;
    delete[] recv_A;
    delete[] send_B;
    delete[] recv_B;
}

