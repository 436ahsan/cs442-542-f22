#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = rand();
    int tag = 1234;
    MPI_Status recv_status;

    if (rank == 0) MPI_Send(&size, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
    if (rank == 1) MPI_Recv(&size, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &recv_status);

    MPI_Finalize();
    return 0;
}


