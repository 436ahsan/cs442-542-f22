#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc <= 1)
    {
        if (rank == 0) printf("Pass global vector dimension n as command line argument\n");
        return 0;
    }

    // Get Global Vector Sizes
    int N = atoi(argv[1]);

    // Split global vector indices across the processes
    // Extra accounts for if num_procs does not evenly divide N
    int n = N / num_procs;
    int extra = N % num_procs;
    int local_n = n;
    int first_n = rank*n;
    if (rank < extra) 
    {
        local_n++;
        first_n += rank;
    }
    else
    {
        first_n += extra;
    }
    

    // Each process gets a portion of the global vector
    double* a = new double[local_n];
    double* b = new double[local_n];
    double* c = new double[local_n];

    // Initialize lists so that process 0 holds
    // a[0] = 0, a[1] = 1, ..., a[n-1] = n-1
    // a[0] = n, a[1] = n+1, ..., a[n-1] = 2n-1
    // ...
    // So list indices stay constant, regardless of number of processes
    for (int i = 0; i < local_n; i++)
    {
        a[i] = first_n + i;
        b[i] = first_n + i;
    }

    // Add local lists together
    for (int i = 0; i < local_n; i++)
        c[i] = a[i] + b[i];

    // Print local result list
    for (int i = 0; i < local_n; i++)
        printf("C[%d] = %e\n", first_n+i, c[i]);


    // Free memory
    delete[] a;
    delete[] b;
    delete[] c;
        
    MPI_Finalize();
    return 0;
}


