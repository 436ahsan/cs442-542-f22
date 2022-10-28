#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

void test_vector()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 2;
    int N = num_procs*n;
    double* A = (double*)malloc(n*N*sizeof(double));
    double* B = (double*)malloc(n*N*sizeof(double));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N+j] = rank*N*n + i*N + j;
            B[i*N+j] = rank;
            printf("Rank %d: A[%d] = %d\n", rank, i*N+j, rank*N*n + i*N + j);
        }
    }

    // Create custom datatype to send column of matrix (stride is N)
    MPI_Datatype newtype;
    MPI_Type_vector(n, 1, N, MPI_DOUBLE, &newtype);
    MPI_Type_commit(&newtype);

    int col = rank;  // Each rank sends a different local column of A
    int proc;
    int even_tag = 1234;
    int odd_tag = 4321;
    if (rank % 2 == 0)
    {
        proc = rank + 1;
        MPI_Send(&(A[col]), 1, newtype, proc, even_tag, MPI_COMM_WORLD);

        //Can recv into different (standard) datatype (holding recvd column as contiguous array)
        MPI_Recv(B, n, MPI_DOUBLE, proc, odd_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
        proc = rank - 1;
        MPI_Recv(B, 1, newtype, proc, even_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&(A[col]), 1, newtype, proc, odd_tag, MPI_COMM_WORLD);
    }
    for (int i = 0; i < N*n; i++)
    {
        printf("Rank %d B[%d] = %e\n", rank, i, B[i]);
    }

    MPI_Type_free(&newtype);
}

void test_contig()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int n = 2;
    int N = num_procs;
    double* A = (double*)malloc(n*N*sizeof(double));
    double* B = (double*)malloc(n*N*sizeof(double));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*n+j] = rank*N*n + i*N + j;
            B[i*n+j] = rank;
            printf("Rank %d: A[%d] = %d\n", rank, i*N+j, rank*N*n + i*N + j);
        }
    }

    // Create custom datatype to send row of matrix (contiguous)
    MPI_Datatype newtype;
    MPI_Type_contiguous(N, MPI_DOUBLE, &newtype);
    MPI_Type_commit(&newtype);

    // Each process sends entire local submatrix, row-wise
    int proc;
    int even_tag = 1234;
    int odd_tag = 4321;
    if (rank % 2 == 0)
    {
        proc = rank + 1;
        MPI_Send(A, n, newtype, proc, even_tag, MPI_COMM_WORLD);
        MPI_Recv(B, n, newtype, proc, odd_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
        proc = rank - 1;
        MPI_Recv(B, n, newtype, proc, even_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(A, n, newtype, proc, odd_tag, MPI_COMM_WORLD);
    }

    for (int i = 0; i < N*n; i++)
        printf("Rank %d, B[%d] = %e\n", rank, i, B[i]);

    MPI_Type_free(&newtype);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test vector custom datatype (constant stride)
    //test_vector();

    // Test contiguous custom datatype 
    test_contig();

    

    MPI_Finalize();
    return 0;
}
