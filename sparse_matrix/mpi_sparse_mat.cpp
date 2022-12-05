#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

#include "../timer.h"
#include "par_binary_IO.hpp"
#include "mpi_sparse_mat.hpp"


void spmv(Mat& A, double* x, double* b)
{
    int start, end, col;
    double val;

    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        for (int j = start; j < end; j++)
        {
            col = A.col_idx[j];
            val = A.data[j];

            b[i] += val * x[col];
        }
    }
}

void form_comm(ParMat& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Gather first row for all processes into list
    std::vector<int> first_rows(num_procs+1);
    MPI_Allgather(&A.first_row, 1, MPI_INT, first_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_rows[num_procs] = A.global_rows;

    // Step through off_proc_columns and find which process the corresponding row is stored on
    std::vector<int> col_to_proc(A.off_proc_num_cols);
    int proc = 0;
    int prev_proc = -1;
    std::vector<int> sizes(num_procs);


    for (int i = 0; i < A.off_proc_num_cols; i++)
    {
        int global_col = A.off_proc_columns[i];
        while (first_rows[proc+1] < global_col)
            proc++;
        col_to_proc[i] = proc;
        if (proc != prev_proc)
        {
            A.recv_comm.procs.push_back(proc);
            A.recv_comm.ptr.push_back(i);
            prev_proc = proc;
            sizes[proc] = 1;
        }
    }
    A.recv_comm.ptr.push_back(A.off_proc_num_cols);
    A.recv_comm.n_msgs = A.recv_comm.procs.size();
    A.recv_comm.req.resize(A.recv_comm.n_msgs);

    // Reduce NSends to Each Proc
    MPI_Allreduce(MPI_IN_PLACE, sizes.data(), num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    A.send_comm.n_msgs = sizes[rank];

    int msg_tag = 1234;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        int start = A.recv_comm.ptr[i];
        int end = A.recv_comm.ptr[i+1];
        MPI_Isend(&(A.off_proc_columns[start]), end - start, MPI_INT, proc, msg_tag, 
                MPI_COMM_WORLD, &(A.recv_comm.req[i]));
    }

    MPI_Status recv_status;
    std::vector<int> recv_buf;
    int count_sum = 0;
    int count;
    A.send_comm.ptr.push_back(0);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, msg_tag, MPI_COMM_WORLD, &recv_status);
        proc = recv_status.MPI_SOURCE;
        A.send_comm.procs.push_back(proc);
        MPI_Get_count(&recv_status, MPI_INT, &count);
        count_sum += count;
        A.send_comm.ptr.push_back(count_sum);
        if (recv_buf.size() < count) recv_buf.resize(count);
        MPI_Recv(recv_buf.data(), count, MPI_INT, proc, msg_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++)
            A.send_comm.idx.push_back(recv_buf[i] - A.first_row);
    }
    A.send_comm.req.resize(A.send_comm.n_msgs);

    MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
}


void graph_create(ParMat& A)
{
    int reorder = 0;
    A.recv_comm.sizes.resize(A.recv_comm.n_msgs);
    A.send_comm.sizes.resize(A.send_comm.n_msgs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
        A.recv_comm.sizes[i] = A.recv_comm.ptr[i+1] - A.recv_comm.ptr[i];
    for (int i = 0; i < A.send_comm.n_msgs; i++)
        A.send_comm.sizes[i] = A.send_comm.ptr[i+1] - A.send_comm.ptr[i];

    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, A.recv_comm.n_msgs, 
            A.recv_comm.procs.data(), A.recv_comm.sizes.data(),
            A.send_comm.n_msgs, A.send_comm.procs.data(), 
            A.send_comm.sizes.data(), MPI_INFO_NULL, reorder,
            &(A.dist_graph_comm));
}

void communicate(ParMat& A, std::vector<double>& data, std::vector<double>& recvbuf)
{
    int proc, start, end;
    int tag = 2948;
    std::vector<double> sendbuf(A.send_comm.idx.size());
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end = A.send_comm.ptr[i+1];
        for (int j = start; j < end; j++)
        {
            sendbuf[j] = data[A.send_comm.idx[j]];
        }
        MPI_Isend(&(sendbuf[start]), end - start, MPI_DOUBLE, proc, tag, 
                MPI_COMM_WORLD, &(A.send_comm.req[i]));
    }

    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end = A.recv_comm.ptr[i+1];
        MPI_Irecv(&(recvbuf[start]), end - start, MPI_DOUBLE, proc, tag,
                MPI_COMM_WORLD, &(A.recv_comm.req[i]));
    }

    MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
}

void par_spmv(ParMat& A, std::vector<double>& x, std::vector<double>& b,
       std::vector<double>& x_dist)
{
    std::fill(b.begin(), b.end(), 0);

    // SpMV with Local Data
    spmv(A.on_proc, x.data(), b.data());

    // Communicate x_values
    communicate(A, x, x_dist);

    spmv(A.off_proc, x_dist.data(), b.data());
}

void par_graph_spmv(ParMat& A, std::vector<double>& x, std::vector<double>& b,
       std::vector<double>& x_dist)
{
    std::fill(b.begin(), b.end(), 0);

    // SpMV with Local Data
    spmv(A.on_proc, x.data(), b.data());

    // Communicate x_values
    MPI_Neighbor_alltoallv(x.data(), A.send_comm.sizes.data(), A.send_comm.ptr.data(), 
            MPI_DOUBLE, x_dist.data(), A.recv_comm.sizes.data(), A.recv_comm.ptr.data(),
            MPI_DOUBLE, A.dist_graph_comm);

    spmv(A.off_proc, x_dist.data(), b.data());
}




int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParMat A;
    char* filename = argv[1];
    double t0, tfinal;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    readParMatrix(filename, A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("I/O Time : %e\n", t0);
    
    /*
    form_comm(A);
    graph_create(A);

    std::vector<double> x(A.local_rows, 1.0);
    std::vector<double> b(A.local_rows);
    std::vector<double> b_graph(A.local_rows);
    std::vector<double> x_dist(A.off_proc_num_cols);

    par_spmv(A, x, b, x_dist);
    par_graph_spmv(A, x, b_graph, x_dist);
    for (int i = 0; i < A.local_rows; i++)
        if (b[i] != b_graph[i])
            printf("B[%d] != BGRAPH!\n", i);


    int n_iter = 10000;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        par_spmv(A, x, b, x_dist);
    }
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Par SpMV Time %e\n", t0);


    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        par_graph_spmv(A, x, b_graph, x_dist);
    }
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Par Graph SpMV Time %e\n", t0);

    */

    MPI_Finalize();
}
