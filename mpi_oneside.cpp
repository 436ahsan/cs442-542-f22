#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void win_create(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* array = (int*)malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        array[i] = rank*n+i;

    MPI_Win win;
    // int MPI_Win_create(void* base, MPI_Aint size, int displ_unit, MPI_Info info, MPI_Comm comm, MPI_Win* win);
    MPI_Win_create(array, n*sizeof(int), sizeof(int),
            MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    int val = rank*n;
    int proc = rank + 1;
    if (rank % 2) proc = rank - 1;


    if (0)
    {
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        MPI_Put(&val, 1, MPI_INT, proc, 0*sizeof(int), 1, MPI_INT, win);
        MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, win);
        printf("Rank %d, array[0] = %d\n", rank, array[0]);
    }
    else if (0)
    {
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);
        MPI_Get(&val, 1, MPI_INT, proc, 0*sizeof(int), 1, MPI_INT, win);
        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        printf("Rank %d, val %d\n", rank, val);
    }
    else if (0)
    {
        MPI_Group group_world;
        MPI_Comm_group(MPI_COMM_WORLD, &group_world);
        MPI_Win_post(group_world, MPI_MODE_NOPUT, win);
        MPI_Win_start(group_world, 0, win);
        MPI_Get(&val, 1, MPI_INT, proc, 0*sizeof(int), 1, MPI_INT, win);
        MPI_Win_complete(win);
        MPI_Win_wait(win);
        MPI_Group_free(&group_world);
        printf("Rank %d, val %d\n", rank, val);
    }
    else if (0)
    {
        MPI_Group group_world, neigh_group;
        MPI_Comm_group(MPI_COMM_WORLD, &group_world);
        int group_size = 2;
        int ranks[2];
        ranks[0] = (rank / 2) * 2;
        ranks[1] = (rank / 2) * 2 + 1;
        MPI_Group_incl(group_world, group_size, ranks, &neigh_group);
        MPI_Win_post(neigh_group, MPI_MODE_NOPUT, win);
        MPI_Win_start(neigh_group, 0, win);
        MPI_Get(&val, 1, MPI_INT, proc, 0*sizeof(int), 1, MPI_INT, win);
        MPI_Win_complete(win);
        MPI_Win_wait(win);
        MPI_Group_free(&neigh_group);
        MPI_Group_free(&group_world);
        printf("Rank %d, val %d\n", rank, val);
    }
    else
    {
        MPI_Group group_world, from_neighbors, to_neighbors;
        MPI_Comm_group(MPI_COMM_WORLD, &group_world);
        int group_size = 2;
        int ranks[2];

        ranks[0] = rank;
        ranks[1] = rank-1;
        if (ranks[1] < 0) ranks[1] += num_procs;
        MPI_Group_incl(group_world, group_size, ranks, &to_neighbors);
        
        MPI_Win_post(to_neighbors, MPI_MODE_NOPUT, win);

        ranks[0] = rank+1;
        if (ranks[0] == num_procs) ranks[0] -= num_procs;
        ranks[1] = rank;
        MPI_Group_incl(group_world, group_size, ranks, &from_neighbors);
        
        MPI_Win_start(from_neighbors, 0, win);
        MPI_Get(&val, 1, MPI_INT, proc, 0*sizeof(int), 1, MPI_INT, win);
        MPI_Win_complete(win);

        MPI_Win_wait(win);

        MPI_Group_free(&from_neighbors);
        MPI_Group_free(&to_neighbors);
        MPI_Group_free(&group_world);
        printf("Rank %d, val %d\n", rank, val);
    }



    MPI_Win_free(&win);
    free(array);
}

void win_allocate(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* array;
    MPI_Win win;
    
    MPI_Win_allocate(n*sizeof(int), sizeof(int), MPI_INFO_NULL, 
            MPI_COMM_WORLD, &array, &win);

    for (int i = 0; i < n; i++)
        array[i] = rank*n+i;

    MPI_Win_free(&win);
}

void win_dynamic(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* array;
    MPI_Win win;

    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    array = (int*)malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        array[i] = rank*n+i;

    MPI_Win_attach(win, array, n*sizeof(int));

    MPI_Win_detach(win, array);

    free(array);
    MPI_Win_free(&win);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = atoi(argv[1]);

    win_create(n);

    return MPI_Finalize();
}
