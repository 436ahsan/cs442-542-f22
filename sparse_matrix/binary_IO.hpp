#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#define PETSC_MAT_CODE 1211216

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>


int readMatrix(const char* filename, int** rowptr, int** col_idx, double** data);

#endif


