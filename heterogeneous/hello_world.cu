#include <stdio.h>
#include <cuda.h>
#include <mpi.h>

__global__ void print_kernel(int rank, int gpu, int node)
{
    printf("Hello from MPI Rank %d on Node %d, GPU %d: block %d, thread %d\n", rank, node, gpu, blockIdx.x, threadIdx.x);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    cudaError_t err;

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);

    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);

    if (rank == 0) printf("Num Nodes %d, Num GPUS %d\n", num_procs / ppn, num_gpus);

    MPI_Comm_free(&node_comm);

    int procs_per_gpu = ppn / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    cudaSetDevice(gpu);

    print_kernel<<<1,1>>>(rank, gpu, rank / ppn);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

    return MPI_Finalize();
}
