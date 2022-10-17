#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mpi.h"
#include <iostream>
#include <string>
#include "mpi_sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "matrix_market.hpp"

void read_mtx_file()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char* filename = "suitesparse/Dubcova2.mtx";
    FILE* f = fopen(filename, "r");
    //FILE* ifile = fopen(filename, 'rb');
    
    MM_typecode matcode;
    mm_read_banner(f, &matcode);
    
    int M, N, nz;
    mm_read_mtx_crd_size(f, &M, &N, &nz);


    int local_nz = nz / num_procs;
    int extra = nz % num_procs;
    int first = local_nz * rank;
    if (rank < extra)
    {
        local_nz++;
        first += rank;
    }
    else
    {
        first += extra;
    }

    std::vector<int> rows(nz);
    std::vector<int> cols(nz);
    std::vector<double> vals(nz);
    int row, col;
    double val;

    for (int i = 0; i < nz; i++)
    {
        int n_items_read = fscanf(f, "%d %d %lg\n", &row, &col, &val);
        row--;
        col--;
        if (row >= first && row < first + local_nz)
        {
            rows.push_back(row);
            cols.push_back(col);
            vals.push_back(val);
        }
    }
    fclose(f);
}


void read_mtx_file_split()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string filename = "suitesparse/Dubcova2_";
    filename += std::to_string(rank);
    filename += ".mtx";
    FILE* f = fopen(filename.c_str(), "r");
    
    MM_typecode matcode;
    mm_read_banner(f, &matcode);
    
    int M, N, nz;
    mm_read_mtx_crd_size(f, &M, &N, &nz);

    std::vector<int> rows(nz);
    std::vector<int> cols(nz);
    std::vector<double> vals(nz);
    int row, col;
    double val;

    for (int i = 0; i < nz; i++)
    {
        int n_items_read = fscanf(f, "%d %d %lg\n", &row, &col, &val);
        row--;
        col--;
        rows.push_back(row);
        cols.push_back(col);
        vals.push_back(val);
    }
    fclose(f);
}

void read_binary_file()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char* filename = "suitesparse/Dubcova2.pm";

    ParMat A;
    readParMatrix(filename, A);
    
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;

    t0 = MPI_Wtime();
    read_mtx_file();
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MTX Read Time %e\n", t0);

    t0 = MPI_Wtime();
    read_mtx_file_split();
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MTX Read Split Time %e\n", t0);

    t0 = MPI_Wtime();
    read_binary_file();
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Binary Read Time %e\n", t0);

    
    return MPI_Finalize();
}
