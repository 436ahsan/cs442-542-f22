#ifndef SPARSE_MAT_HPP
#define SPARSE_MAT_HPP

#include <vector>

struct Mat 
{
    std::vector<int> rowptr;
    std::vector<int> col_idx;
    std::vector<double> data;
    int n_rows;
    int n_cols;
    int nnz;
};

#endif

