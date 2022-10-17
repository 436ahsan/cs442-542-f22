#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

#include "../timer.h"
#include "binary_IO.hpp"


// Struct for sparse matrix object
struct Mat 
{
    std::vector<int> rowptr;
    std::vector<int> col_idx;
    std::vector<double> data;
    int n;
    int nnz;
};

// Multiplies two sparse matrices A*B into a third sparse matrix C
// Assumes all square matrices
void spgemm(Mat& A, Mat& B, Mat& C)
{
    int start_A, end_A;
    int start_B, end_B;
    int col_A, col_B, col_C;
    double val_A, val_B;

    int n_cols;
    std::vector<int> col_idx(A.n);
    double* col_sums = new double[A.n];
    for (int i = 0; i < A.n; i++)
        col_sums[i] = 0;

    C.n = A.n;
    C.rowptr.push_back(0);
    for (int i = 0; i < A.n; i++)
    {
        // Go through a single row of A
        start_A = A.rowptr[i];
        end_A = A.rowptr[i+1];
        n_cols = 0;
        for (int j = start_A; j < end_A; j++)
        {
            // Go through rows of B corresponding to column index in row i of A
            col_A = A.col_idx[j];
            val_A = A.data[j];
            start_B = B.rowptr[col_A];
            end_B = B.rowptr[col_A+1];
            for (int k = start_B; k < end_B; k++)
            {
                col_B = B.col_idx[j];
                val_B = B.data[j];

                // If nothing in column yet, add to col_idx
                if (fabs(col_sums[col_B]) < 1e-100)
                {
                    col_idx[n_cols++] = col_B;
                }

                // Add to col_sums variable
                col_sums[col_B] += val_B * val_A;
            }
        }

        for (int j = 0; j < n_cols; j++)
        {
            // Add column index and data value to C
            col_C = col_idx[j];
            C.col_idx.push_back(col_C);
            C.data.push_back(col_sums[col_C]);

            // Reset col_sums for next row of A
            col_sums[col_C] = 0;
        }
        C.rowptr.push_back(C.col_idx.size());
    }
}


// Multiplies two sparse matrices A*B into a third sparse matrix C
// Assumes all square matrices
void spgemm_inefficient(Mat& A, std::vector<int>& B_colptr, std::vector<int>& B_rows, std::vector<double>& B_data, Mat& C)
{
    int idx_A, end_A;
    int idx_B, end_B, row_B;
    int col_A, col_B, col_C;
    double val_A, val_B, sum;

    int n_cols;

    C.n = A.n;
    C.rowptr.push_back(0);
    
    // Go through a single row of A
    for (int i = 0; i < A.n; i++)
    {
        idx_A = A.rowptr[i];
        end_A = A.rowptr[i+1];
        // Go through a single column of B
        for (int j = 0; j < A.n; j++)
        {
            sum = 0;
            idx_B = B_colptr[j];
            end_B = B_colptr[j+1];
            // Go through row * column
            while (idx_A < end_A || idx_B < end_B)
            {
                if (idx_A < end_A)
                    col_A = A.col_idx[idx_A];
                else col_A = A.n;

                if (idx_B < end_B)
                    row_B = B_rows[idx_B];
                else col_B = A.n;

                if (col_A == row_B)
                {
                    sum += A.data[idx_A] * B_data[idx_B];
                    idx_A++;
                    idx_B++;
                }
                else if (col_A < col_B)
                    idx_A++;
                else idx_B++;
                
            }
            if (fabs(sum) > 0)
            {
                C.col_idx.push_back(j);
                C.data.push_back(sum);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    int* idx1;
    int* idx2;
    double* data;
    int start, end, col;
    double val;

    if (argc == 1) 
    {
        printf("Add filename to commandline\n");
        return 0;
    }

    // Read in matrix
    char* filename = argv[1];
    int n_rows = readMatrix(filename, &idx1, &idx2, &data);
    int nnz = idx1[n_rows];

    // Create three structs
    struct Mat A;
    struct Mat B;
    struct Mat C;

    // Multiplying matrix by itself (computing A^2)
    // Add matrix indices and values from readMatrix function 
    // to both A and B
    A.n = n_rows;
    B.n = n_rows;
    A.rowptr.push_back(0);
    B.rowptr.push_back(0);
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = idx2[j];
            val = data[j];

            A.col_idx.push_back(col);
            A.data.push_back(val);
            B.col_idx.push_back(col);
            B.data.push_back(val);
        }
        A.rowptr.push_back(A.col_idx.size());
        B.rowptr.push_back(B.col_idx.size());
    }
    A.nnz = A.rowptr[A.n];
    B.nnz = B.rowptr[B.n];


    // Multiply A*B
    int n_iter = 10;
    double start_t = get_time();
    for (int i = 0; i < n_iter; i++)
    { 
        spgemm(A, B, C);
    }
    double end_t = get_time();
    printf("SpGEMM took %e seconds\n", (end_t - start_t) / n_iter);


    // Create CSC Version of B (to mult row of A by col of B)
    int idx;
    std::vector<int> colptr(n_rows+1); // square matrix
    std::vector<int> rows(B.nnz);
    std::vector<double> B_data(B.nnz);
    std::vector<int> col_sizes(n_rows, 0);
    for (int i = 0; i < B.nnz; i++)
    {
        col_sizes[B.col_idx[i]]++;
    }
    colptr[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        colptr[i+1] = colptr[i] + col_sizes[i];
        col_sizes[i] = 0;
    }
    for (int i = 0; i < n_rows; i++)
    {
        start = B.rowptr[i];
        end = B.rowptr[i+1];
        for (int j = start; j < end; j++)
        {
            col = B.col_idx[j];
            idx = col_sizes[col]++;
            rows[idx] = i;
            B_data[idx] = B.data[j];
        }
    }

    // Multiply rows of A by cols of B
    n_iter = 1;
    start_t = get_time();
    for (int i = 0; i < n_iter; i++)
    { 
        spgemm_inefficient(A, colptr, rows, B_data, C);
    }
    end_t = get_time();
    printf("Inefficient SpGEMM took %e seconds\n", (end_t - start_t) / n_iter);


    delete[] idx1;
    delete[] idx2;
    delete[] data;

    return 0;
}


