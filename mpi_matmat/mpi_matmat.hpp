#ifndef MPI_MATMAT_HPP
#define MPI_MATMAT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

// Return process in process-row 'row' and
// process-column 'col'
int get_proc(int row, int col, int sq_procs);

// Performs local (serial)matrix-multiplication
void matmat(int n, double* A, double* B, double* C);

// Calculates sum of matrix (for checking solutions are likely same)
double mat_sum(int n, double* C);


// Simplest implementation of parallel DGEMM
//     Gathers entires rows of A, cols of B
//     Method in 'simple.cpp'
void mpi_matmat_simple(double* A, double* B, double* C, 
        int n, int sq_num_procs,int rank_row, int rank_col);

// Fox's Algorithm
//     Gathers entire rows of A
//     Rotates chunks of B down the column
//     Method in 'fox.cpp'
void mpi_matmat_fox(double* A, double* B, double* C, 
        int n, int sq_num_procs,int rank_row, int rank_col);

// Cannon's Algorithm; To Be Written By You
//     Rotates chunks of A right through row
//     Rotates chunks of B down column
//     Add this method to 'cannon.cpp'
void mpi_matmat_cannon(double* A, double* B, double* C, 
        int n, int sq_num_procs,int rank_row, int rank_col);

#endif
