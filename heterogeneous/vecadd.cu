#include <cuda.h>
#include <mpi.h>
#include "../timer.h"

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(float* A, float* B, float* C, int n)
{
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

double sum(int n, float* h_C)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += h_C[i];
    return s;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, 
            MPI_INFO_NULL, &node_comm);

    int node_rank, ppn;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);

    int procs_per_gpu = ppn / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    cudaSetDevice(gpu);

    MPI_Comm_free(&node_comm);

    float *h_A, *h_B, *h_C;
    double t0, tfinal;

    int N = atoi(argv[1]);
    int n = N / num_procs;
    int extra = N % num_procs;
    if (rank < extra) n++;

    h_A = (float*)malloc(n*sizeof(float));
    h_B = (float*)malloc(n*sizeof(float));
    h_C = (float*)malloc(n*sizeof(float));

    for (int i = 0; i < n; i++)
    {
        h_A[i] = 0.5;
        h_B[i] = 0.7;
    }

    t0 = get_time();
    vecAdd(h_A, h_B, h_C, n);
    tfinal = get_time() - t0;
    double sum_C = sum(n, h_C);
    double sum_t;
    MPI_Reduce(&sum_C, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("VecAdd Time %e, Sum %e\n", t0, sum_t);


    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n*sizeof(float));
    cudaMalloc((void**)&d_B, n*sizeof(float));
    cudaMalloc((void**)&d_C, n*sizeof(float));

    // Copy host array to device array
    t0 = get_time();
    cudaMemcpy(d_A, h_A, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n*sizeof(float), cudaMemcpyHostToDevice);

    // Launch GPU Kernel
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, n*sizeof(float), cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;
    
    sum_C = sum(n, h_C);
    MPI_Reduce(&sum_C, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("VecAddKernel Time %e, Sum %e\n", t0, sum_t);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return MPI_Finalize();
}
