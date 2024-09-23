// This will be a program to do matrix addition using CUDA and on the CPU and it will compare the results.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void init_matrix(int *A, int *B, int N);
void add_matrix_cpu(int *A, int *B, int *C, int N);
__global__ void add_matrix_gpu(int *A, int *B, int *C, int N);
void compare_matrices(int *cpu_result, int *gpu_result, int N);

void main(){
    int N, block_size;

    // Get Matrix size from user
    printf("Enter size of the N x N matrix: ");
    scanf("%d", &N);

    // Get bloack size from user
    printf("enter the block size for CUDA: ");
    scanf("%d", &block_size);

    // Allocate memory for matrices
    int *A = (int *)malloc(N * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));
    int *C_cpu = (int *)malloc(N * N * sizeof(int));
    int *C_gpu = (int *)malloc(N * N * sizeof(int));

    init_matrix(A, B, N);
    add_matrix_cpu(A, B, C_cpu, N);

    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    cudeMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    add_matrix_gpu<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C_gpu, d_c, N * N * sizeof(int), cudeMemcpyDeviceToHost);

    compare_matrices(C_cpu, C_gpu, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    
    return 0;
}

void init_matrix(int *A, int *B, int N){
    int init=1325;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            init= (3125 * init) % 65536;
            A[i * N + j] = (init - 32768) / 6553;
            B[i * N + j] = init % 1000;
        }
    }
}

void add_matrix_cpu(int *A, int *B, int *C, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            int index = i * N + j;
            C[index] = A[index] + B[index];
        }
    }
}

__global__ void add_matrix_gpu(int *A, int *B, int *C, int N){
    int col = blockIDx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * N + col;

    if( row < N && col < N){
        c[index] = a[index] + b[index];
    }
}

void compare_matrices(int *cpu_result, int *gpu_result, int N){
    for(int i = 0; i < N; i++){
        if(cpu_result[i] != gpu_result[i]){
            printf("Mismatch at index %d! CPU: %d, GPU: %d\n", i, cpu_result[i], gpu_result[i]);
            return;
        }
    }
    printf("CPU and GPU results match!\n");
}