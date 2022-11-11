#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "timer.h"

// To compile with and without vectorization (in gcc):
// gcc -o <executable_name> <file_name> -O1     <--- no vectorization
// Flag to vectorize : -ftree-vectorize  
// Flag needed for vectorization of X86 processors : -msse -msse2
// Flag needed for vectorization of PowerPC platforms : -maltivec
// Other optional flags (floating point reductions) : -ffast-math -fassociative-math
//
// To see what the compiler vectorizes : -fopt-info-vec (or -fopt-info-vec-optimized)
// To see what the compiler is not able to vectorize : -fopt-info-vec-missed


// Matrix-Matrix Multiplication of Doubles (Double Pointer)
// Test without the restrict variables
void matmat(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
                C[i*n+k] = 0;

            for (int j = 0; j < n; j++)
            {
                val = A[i*n+j];
                for (int k = 0; k < n; k++)
                {
                    C[i*n+k] += val * B[j*n+k];
                }
            }
        }
    }
}



void matmat_unrolledi_ik(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val2, val3, val4, bval;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i += 4)
        {
            // Initialize entries of C rows [i:i+4) to 0
            for (int k = 0; k < n; k++)
            {
                C[i*n+k] = 0;
                C[(i+1)*n+k] = 0;
                C[(i+2)*n+k] = 0;
                C[(i+3)*n+k] = 0;
            }

            // Multiple four rows [i:i+4) of A by columns of B
            for (int j = 0; j < n; j++)
            {
                val = A[i*n+j];
                val2 = A[(i+1)*n+j];
                val3 = A[(i+2)*n+j];
                val4 = A[(i+3)*n+j];
                for (int k = 0; k < n; k++)
                {
                    bval = B[j*n+k];
                    C[i*n+k] += val * bval;
                    C[(i+1)*n+k] += val2 * bval;
                    C[(i+2)*n+k] += val3 * bval;
                    C[(i+3)*n+k] += val4 * bval;
                }
            }
        }
    }
}


void matmat_unrolledi_ij(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val2, val3, val4, bval;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i += 4)
        {
            // Multiple four rows [i:i+4) of A by columns of B
            for (int j = 0; j < n; j++)
            {
                val = 0;
                val2 = 0;
                val3 = 0;
                val4 = 0;
                for (int k = 0; k < n; k++)
                {
                    bval = B[k*n+j];
                    val += A[i*n+k] * bval;
                    val2 += A[(i+1)*n+k] * bval;
                    val3 += A[(i+2)*n+k] * bval;
                    val4 += A[(i+3)*n+k] * bval;
                }
                C[i*n+j] = val;
                C[(i+1)*n+j] = val2;
                C[(i+2)*n+j] = val3;
                C[(i+3)*n+j] = val4;
            }
        }
    }
}



void matmat_unrolledi_ki(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val2, val3, val4, aval;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i += 4)
        {
            // Initialize entries of C rows [i:i+4) to 0
            for (int k = 0; k < n; k++)
            {
                C[k*n+i] = 0;
                C[k*n+i+1] = 0;
                C[k*n+i+2] = 0;
                C[k*n+i+3] = 0;
            }

            // Multiple four rows [i:i+4) of A by columns of B
            for (int j = 0; j < n; j++)
            {
                val = B[j*n+i];
                val2 = B[j*n+i+1];
                val3 = B[j*n+i+2];
                val4 = B[j*n+i+3];
                for (int k = 0; k < n; k++)
                {
                    aval = A[k*n+j];
                    C[k*n+i] += aval * val;
                    C[k*n+i+1] += aval * val2;
                    C[k*n+i+2] += aval * val3;
                    C[k*n+i+3] += aval * val4;
                }
            }
        }
    }
}


void matmat_unrolledi_kj(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val2, val3, val4;
    for (int iter = 0; iter < n_iter; iter++)
    {        
        for (int i = 0; i < n*n; i++)
            C[i] = 0;
        for (int i = 0; i < n; i += 4)
        {
            // Multiple four rows [i:i+4) of A by columns of B
            for (int j = 0; j < n; j++)
            {
                val = B[i*n+i];
                val2 = B[(i+1)*n+j];
                val3 = B[(i+2)*n+j];
                val4 = B[(i+3)*n+j];
                for (int k = 0; k < n; k++)
                {
                    C[k*n+j] += A[k*n+i] * val;
                    C[k*n+j] += A[k*n+i+1] * val2;
                    C[k*n+j] += A[k*n+i+2] * val3;
                    C[k*n+j] += A[k*n+i+3] * val4;
                }
            }
        }
    }
}


void matmat_unrolledi_ji(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val2, val3, val4, aval;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i += 4)
        {
            // Multiple four rows [i:i+4) of A by columns of B
            for (int j = 0; j < n; j++)
            {
                val = 0;
                val2 = 0;
                val3 = 0;
                val4 = 0;
                for (int k = 0; k < n; k++)
                {
                    aval = A[j*n+k];
                    val += aval*B[k*n+i];
                    val2 += aval*B[k*n+i+1];
                    val3 += aval*B[k*n+i+2];
                    val4 += aval*B[k*n+i+3];
                }
                C[j*n+i] = val;
                C[j*n+i+1] = val2;
                C[j*n+i+2] = val3;
                C[j*n+i+3] = val4;

            }
        }
    }
}



void matmat_unrolledi_jk(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val2, val3, val4, aval;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n*n; i++)
            C[i] = 0;

        for (int i = 0; i < n; i += 4)
        {
            // Multiple four rows [i:i+4) of A by columns of B
            for (int j = 0; j < n; j++)
            {
                val = A[j*n+i];
                val2 = A[j*n+i+1];
                val3 = A[j*n+i+2];
                val4 = A[j*n+i+3];

                for (int k = 0; k < n; k++)
                {
                    C[j*n+k] += val*B[i*n+k];
                    C[j*n+k] += val2*B[(i+1)*n+k];
                    C[j*n+k] += val3*B[(i+2)*n+k];
                    C[j*n+k] += val4*B[(i+3)*n+k];
                }
            }
        }
    }
}

void matmat_unrolledj_ik(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val1, val2, val3;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
                C[i*n+k] = 0;

            for (int j = 0; j < n; j += 4)
            {
                val = A[i*n+j];
                val1 = A[i*n+j+1];
                val2 = A[i*n+j+2];
                val3 = A[i*n+j+3];
                for (int k = 0; k < n; k++)
                {
                    C[i*n+k] += val * B[j*n+k];
                    C[i*n+k] += val1 * B[(j+1)*n+k];
                    C[i*n+k] += val2 * B[(j+2)*n+k];
                    C[i*n+k] += val3 * B[(j+3)*n+k];
                }
            }
        }
    }
}

void matmat_unrolledj_ij(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val1, val2, val3;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j += 4)
            {
                C[i*n+j] = 0;
                C[i*n+j+1] = 0;
                C[i*n+j+2] = 0;
                C[i*n+j+3] = 0;
                for (int k = 0; k < n; k++)
                {
                    val = A[i*n+k];
                    C[i*n+j] += val * B[k*n+j];
                    C[i*n+j+1] += val * B[k*n+j+1];
                    C[i*n+j+2] += val * B[k*n+j+2];
                    C[i*n+j+3] += val * B[k*n+j+3];
                }
            }
        }
    }
}

void matmat_unrolledj_ji(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val1, val2, val3;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j += 4)
            {
                C[j*n+i] = 0;
                C[(j+1)*n+i] = 0;
                C[(j+2)*n+i] = 0;
                C[(j+3)*n+i] = 0;
                for (int k = 0; k < n; k++)
                {
                    val = B[k*n+i];
                    C[j*n+i] += A[j*n+k] * val;
                    C[(j+1)*n+i] += A[(j+1)*n+k] * val;
                    C[(j+2)*n+i] += A[(j+2)*n+k] * val;
                    C[(j+3)*n+i] += A[(j+3)*n+k] * val;
                }
            }
        }
    }
}

void matmat_unrolled(int n, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int n_iter)
{
    double val, val1, val2, val3;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j += 4)
            {
                C[j*n+i] = 0;
                C[(j+1)*n+i] = 0;
                C[(j+2)*n+i] = 0;
                C[(j+3)*n+i] = 0;
                for (int k = 0; k < n; k++)
                {
                    val = B[k*n+i];
                    C[j*n+i] += A[j*n+k] * val;
                    C[(j+1)*n+i] += A[(j+1)*n+k] * val;
                    C[(j+2)*n+i] += A[(j+2)*n+k] * val;
                    C[(j+3)*n+i] += A[(j+3)*n+k] * val;
                }
            }
        }
    }
}



// This program runs matrix matrix multiplication with double pointers
// Test vectorization improvements for both doubles and floats
// Try with and without the restrict variables
int main(int argc, char* argv[])
{

    double start, end;
    int n_access = 1000000000;

    if (argc < 2)
    {
        printf("Need Matrix Dimemsion n and step size k passed as Command Line Arguments (e.g. ./matmat 8 2)\n");
        return 0;
    }

    int n = atoi(argv[1]);
    int step = atoi(argv[2]);

    int n_iter = (n_access / (n*n*n)) + 1;

    double* A = (double*)malloc(n*n*sizeof(double));
    double* B = (double*)malloc(n*n*sizeof(double));
    double* C = (double*)malloc(n*n*sizeof(double));
    double* C_new = (double*)malloc(n*n*sizeof(double));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i*n+j] = 1.0/(i+1);
            B[i*n+j] = 1.0;
        }
    }


    // Comparisons
    matmat(n, A, B, C, 2);
    matmat_unrolled(n, A, B, C_new, 2);
    for (int i = 0; i < n*n; i++)
        if (fabs(C[i] - C_new[i]) > 1e-10)
        {
            printf("Different Answers (Unrolled)! idx %d, %e vs %e\n", i, C[i], C_new[i]);
            return 0;
        }

    
    // Warm-Up 
    matmat(n, A, B, C, n_iter);

    start = get_time();
    matmat(n, A, B, C, n_iter);
    end = get_time();
    printf("N %d, Time Per MatMat %e\n", n, (end - start)/n_iter);


    // Warm-Up 
    matmat_unrolled(n, A, B, C_new, 2);

    start = get_time();
    matmat_unrolled(n, A, B, C_new, 2);
    end = get_time();
    printf("N %d, Time Per MatMat Unrolled %e\n", n, (end - start) / n_iter);


    free(A);
    free(B);
    free(C);
        

    return 0;
}
