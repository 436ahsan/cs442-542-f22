#ifndef MPI_MATMAT_HPP
#define MPI_MATMAT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

// Return process in process-row 'row' and
// process-column 'col'
int get_proc(int row, int col, int sq_procs)
{
    return row*sq_procs + col;
}

// Performs local (cuda) matrix-multiplication
__global__ void matmat(float* A, float* B, float* C, int n)

// Calculates sum of matrix (for checking solutions are likely same)
double mat_sum(int n, double* C);

// Cannon's Algorithm; To Be Written By You
//     Rotates chunks of A right through row
//     Rotates chunks of B down column
//     Add this method to 'cannon.cpp'
void mpi_matmat_cannon(double* A, double* B, double* C, 
        int n, int sq_num_procs,int rank_row, int rank_col);

#endif
