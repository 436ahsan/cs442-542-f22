#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

#include "../timer.h"
#include "mat_readers/binary_IO.hpp"
#include "sparse_mat.hpp"

void spmv(int n, int* rowptr, int* col_idx, double* data, double* x, double* b)
{
    int start, end, col;
    double val;

#pragma omp parallel for schedule(runtime) private(start, end, col, val)
    for (int i = 0; i < n; i++)
    {
        b[i] = 0;
        start = rowptr[i];
        end = rowptr[i+1];
        for (int j = start; j < end; j++)
        {
            col = col_idx[j];
            val = data[j];

            b[i] += val * x[col];
        }
    }
}

int main(int argc, char* argv[])
{
    int* idx1;
    int* idx2;
    double* data;

    char* filename = argv[1];
    int n_rows = readMatrix(filename, &idx1, &idx2, &data);
    int nnz = idx1[n_rows];

    double* x = new double[n_rows];
    double* b = new double[n_rows];
    for (int i = 0; i < n_rows; i++)
    {
        x[i] = (double)(rand()) / RAND_MAX;
    }


    int n_iter = 100;
    double start = get_time();
    for (int i = 0; i < n_iter; i++)
    { 
        spmv(n_rows, idx1, idx2, data, x, b);
    }
    double end = get_time();
    printf("SpMV took %e seconds\n", (end - start) / n_iter);



    delete[] x;
    delete[] b;
    delete[] idx1;
    delete[] idx2;
    delete[] data;

    return 0;
}

