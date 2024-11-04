// Brandon Walker
// CS4370
// Parallel Programming Many-Core GPUs
// Meilin Liu
// 4-Nov-2024
// Sum Reduction

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void init_matrix(int *A, int *B, int N);
void SumReduction(int* x, int N);
__global__ void SumReductionKernel(int* x, int N);
void compare_matrices(int *cpu_result, int *gpu_result, int N);
void print_matrix(int *matrix, int N, const char *name);

int main(){
    int Width, block_size;

    // Get Matrix size from user
    cout << "Enter size of the Array: ";
    cin >> Width;

    // Get bloack size from user
    cout << "Enter the block size for CUDA: ";
    cin >> block_size;

    // Allocate memory for matrices
    int *A = new int[Width];
    int *B = new int[Width];

    init_matrix(A, B, Width);

    print_matrix(A, Width, "Matrix A");
    print_matrix(B, Width, "Matrix B");

    auto start_cpu = chrono::high_resolution_clock::now();

    SumReduction(A, Width);

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_cpu = end_cpu - start_cpu;
    cout << "CPU time: " << duration_cpu.count() << " ms" << endl;

    int *d_B;
    cudaMalloc(&d_B, Width * sizeof(int));

    cudaMemcpy(d_B, B, Width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(block_size);
    dim3 dimGrid((Width + block_size - 1) / block_size);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    size_t shared_mem_size = block_size * sizeof(int);

    while(dimGrid > 1){
        SumReductionKernel<<<dimGrid, dimBlock, shared_mem_size>>>(d_B, Width);
        cudaDeviceSynchronize();
        Width = dimGrid;
        dimGrid = (Width + dimBlock - 1) / dimBlock;
    }

    SumReductionKernel<<<1, dimBlock, shared_mem_size>>>(d_B, Width);
    

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    cout << "GPU time: " << milliseconds << " ms" << endl;

    cudaMemcpy(B, d_B, Width * sizeof(int), cudaMemcpyDeviceToHost);

    print_matrix(A, Width, "Matrix A (CPU)");
    print_matrix(B, Width, "Matrix B (GPU)");

    compare_matrices(A, B, Width);

    cudaFree(d_B);

    delete[] A;
    delete[] B;
    
    return 0;
}

void init_matrix(int *A, int *B, int N){
    int init = 1325;
    for(int i = 0; i < N; i++){
        init = 3125 * init % 6553;
        A[i] = (init - 1000) % 97;
        B[i] = (init - 1000) % 97;
    }
}

void SumReduction(int* x, int N){
    for(int i = 1; i < N; i++){
        x[0] += x[i];
    }
}

__global__ void SumReductionKernel(int* x, int Width){
    extern __shared__ int sdata[];

    int threadID = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[threadID] = (i < Width) ? x[i] : 0;
    __syncthreads();

    for(int half = blockDim.x / 2; half > 0; half /=2){
        if(threadID < half){
            sdata[threadID] += sdata[threadID + half];
        }
        __syncthreads();
    }

    if(threadID == 0){
        x[blockIdx.x] = sdata[0];
    }
}

void compare_matrices(int *cpu_result, int *gpu_result, int N){
    for(int i = 0; i < N; i++){
        if(cpu_result[i] != gpu_result[i]){
            cout << "Matrices are not equal" << endl;
            return;
        }
    }
    cout << "Matrices are equal" << endl;
}

void print_matrix(int *matrix, int N, const char *name){
    cout << name << ":" << endl;
    for(int i = 0; i < N; i++){
        cout << matrix[i] << " ";
        if((i + 1) % 10 == 0){
            cout << endl;
        }
    }
    cout << endl;
}