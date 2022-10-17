// Declare private methods
bool little_endian();


#include "binary_IO.hpp"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

bool little_endian()
{
    int num = 1;
    return (*(char *)&num == 1);
}

template <class T>
void endian_swap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}

int readMatrix(const char* filename, int** rowptr, int** col_idx, double** data)
{
    int32_t code;
    int32_t n_rows;
    int32_t n_cols;
    int32_t nnz;
    int32_t idx;
    double val;

    int sizeof_dbl = sizeof(val);
    int sizeof_int32 = sizeof(code);
    bool is_little_endian = false;

    std::ifstream ifs (filename, std::ifstream::binary);
    ifs.read(reinterpret_cast<char *>(&code), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&n_rows), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&n_cols), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&nnz), sizeof_int32);

    if (code != PETSC_MAT_CODE)
    {
        is_little_endian = true;
        endian_swap(&code);
        endian_swap(&n_rows);
        endian_swap(&n_cols);
        endian_swap(&nnz);
    }

    assert(code == PETSC_MAT_CODE);

    int* idx1 = new int[n_rows+1];
    int* idx2 = new int[nnz];
    double* vals = new double[nnz];

    int displ = 0;
    idx1[0] = 0;
    if (is_little_endian)
    {
        for (int32_t i = 0; i < n_rows; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            endian_swap(&idx);
            displ += idx;
            idx1[i+1] = displ;
        }
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            endian_swap(&idx);
            idx2[i] = idx;
        }
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&val), sizeof_dbl);
            endian_swap(&val);
            vals[i] = val;
        }
    }
    else
    {
        for (int32_t i = 0; i < n_rows; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            displ += idx;
            idx1[i+1] = displ;
        }   
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            idx1[i] = idx;
        }
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&val), sizeof_dbl);
            endian_swap(&val);
            vals[i] = val;
        }
    }

    ifs.close();

    *rowptr = idx1;
    *col_idx = idx2;
    *data = vals;

    return n_rows;

}



