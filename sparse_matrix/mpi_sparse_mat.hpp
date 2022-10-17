#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include "mpi.h"
#include "sparse_mat.hpp" 


struct Comm
{
    int n_msgs;
    std::vector<int> procs;
    std::vector<int> ptr;
    std::vector<int> idx;
    std::vector<int> sizes;
    std::vector<MPI_Request> req;
};

struct ParMat
{
    Mat on_proc;
    Mat off_proc;
    int global_rows;
    int global_cols;
    int local_rows;
    int local_cols;
    int first_row;
    int first_col;
    int off_proc_num_cols;
    std::vector<int> off_proc_columns;
    Comm send_comm;
    Comm recv_comm;
    MPI_Comm dist_graph_comm;
};

#endif
