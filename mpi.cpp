#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = atoi(argv[1]);

    int* array = (int*)malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        array[i] = rank*n+i;

    MPI_Win win;
    MPI_Win_create(array, n*sizeof(int), sizeof(int),
            MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_free(&win);
    free(array);

    MPI_Finalize();
    return 0;
}


