#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#define PETSC_MAT_CODE 1211216

#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "mpi_sparse_mat.hpp"
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>


void readParMatrix(const char* filename, ParMat& A);

#endif


