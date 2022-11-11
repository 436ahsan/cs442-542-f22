#include <cuda.h>
#include "../timer.h"

__global__ void matrixMultKernel(float* A, float* B, float* C, int n)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row < n && col < n)
    {
        float val = 0;
        for (int k = 0; k < n; k++)
            val += A[row*n+k] * B[k*n+col];
        C[row*n+col] = val;
    }
}

void matrixMult(float* A, float* B, float* C, int n)
{
    float val;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            val = 0;
            for (int k = 0; k < n; k++)
                val += A[i*n+k] * B[k*n+j];
            C[i*n+j] = val;
        }
    }
}

double sum(float* C, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            s += C[i*n+j];
    return s;
}


int main(int argc, char* argv[])
{
for (int i = 0; i < 1000000; i++)
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    double t0, tfinal;
    
    int n = atoi(argv[1]);
    int size = n*n*sizeof(float);
    
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    for (int i = 0; i < n*n; i++)
    {
        h_A[i] = 0.5;
        h_B[i] = 0.2;
    }   
    
    t0 = get_time();
    matrixMult(h_A, h_B, h_C, n);
    tfinal = get_time() - t0;
    printf("MatrixMult Time %e, Sum %e\n", tfinal, sum(h_C, n));
    

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    

    // Matmat
    dim3 dimBlock(32,32);
    int grid_dim = ceil(n / 32.0);
    dim3 dimGrid(grid_dim, grid_dim);
    
    t0 = get_time();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    matrixMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;
    printf("MatrixMultKernel Time %e, Size %e\n", tfinal, sum(h_C, n));
    

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
}




