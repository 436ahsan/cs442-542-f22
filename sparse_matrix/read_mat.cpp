#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mat_readers/binary_IO.hpp"

int main(int argc, char* argv[])
{
    int* rowptr;
    int* col_idx;
    double* data;

    int start, end, col;
    double val;

    char* filename = argv[1];

    int n_rows = readMatrix(filename, &rowptr, &col_idx, &data);

    for (int i = 0; i < n_rows; i++)
    {
        start = rowptr[i];
        end = rowptr[i+1];
        for (int j = start; j < end; j++)
        {
            col = col_idx[j];
            val = data[j];
            printf("A[%d][%d] = %e\n", i, col, val);
        }
    }

    return 0;
}
