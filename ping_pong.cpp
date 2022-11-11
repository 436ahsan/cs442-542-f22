#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <vector>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n_iter = 10000;
    int max_i = 19;
    int size;
    int max_size = pow(2, max_i-1);
    std::vector<float> vals(max_size);

    int send_tag = 43245;
    int recv_tag = 83243;

    for (int test = 0; test < 5; test++)
    {
        for (int i = 0; i < max_i; i++)
        {
            size = pow(2, i);

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                if (rank == 0)
                {
                    MPI_Send(vals.data(), size, MPI_FLOAT, 1, send_tag, MPI_COMM_WORLD);
                    MPI_Recv(vals.data(), size, MPI_FLOAT, 1, recv_tag, MPI_COMM_WORLD,
                            MPI_STATUSES_IGNORE);
                }
                else
                {
                    MPI_Recv(vals.data(), size, MPI_FLOAT, 0, send_tag, MPI_COMM_WORLD,
                            MPI_STATUSES_IGNORE);
                    MPI_Send(vals.data(), size, MPI_FLOAT, 0, recv_tag, MPI_COMM_WORLD);
                }
            }
            double tfinal = (MPI_Wtime() - t0) / (2*n_iter);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Size %d, Time %e\n", size*sizeof(float), t0);
        }
    }
    MPI_Finalize();
}

