// Brandon Walker
// CS4370
// Parallel Programming Many-Core GPUs
// Meilin Liu
// 4-Nov-2024
// Parallel Prefix Sum

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void init_matrix(int *A, int *B, int Width);
void ParPrefix(int* x, int* y, int Width);
__global__ void ParPrefixKernel(int* x,int* y, int Width);
void compare_matrices(int cpu_result, int gpu_result);
void print_matrix(int *matrix, const char *name);

int main(){
    int Width, block_size;

    // Get Matrix size from user
    cout << "Enter size of the Array: ";
    cin >> Width;

    // Get block size from user
    cout << "Enter the block size for CUDA: ";
    cin >> block_size;

    // Allocate memory for matrices
    int* A = new int[Width];
    int* B = new int[Width];
    int* cpuSum = new int[Width];
    int* gpuSum = new int[Width];

    init_matrix(A, B, Width);

    print_matrix(A, "Matrix A");
    print_matrix(B, "Matrix B");

    auto start_cpu = chrono::high_resolution_clock::now();

    ParPrefix(A, cpuSum, Width);

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_cpu = end_cpu - start_cpu;
    cout << "CPU time: " << duration_cpu.count() << " ms" << endl;

    int* d_B;
    int* d_Sum;
    cudaMalloc(&d_B, Width * sizeof(int));
    cudaMalloc(&d_Sum, Width * sizeof(int));

    cudaEvent_t transfer_start, transfer_stop;
    cudaEventCreate(&transfer_start);
    cudaEventCreate(&transfer_stop);

    cudaEventRecord(transfer_start);
    cudaMemcpy(d_B, B, Width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sum, gpuSum, Width * sizeof(int), cudaMemcpuHostToDevice);

    dim3 dimBlock(block_size);
    dim3 dimGrid((Width + block_size - 1) / block_size);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    size_t shared_mem_size = block_size * sizeof(int);

    ParPrefixKernel<<<dimGrid, dimBlock, shared_mem_size>>>(d_B, d_Sum, Width);

    // while(dimGrid.x > 1){
    //     SumReductionKernel<<<dimGrid, dimBlock, shared_mem_size>>>(d_B, Width);
    //     cudaDeviceSynchronize();
    //     Width = dimGrid.x;
    //     dimGrid.x = (Width + dimBlock.x - 1) / dimBlock.x;
    // }

    // SumReductionKernel<<<1, dimBlock, shared_mem_size>>>(d_B, Width);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cudaMemcpy(B, d_B, Width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuSum, d_Sum, Width * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(transfer_stop);
    cudaEventSynchronize(transfer_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    cout << "GPU time: " << milliseconds << " ms" << endl;

    float mem_transfer = 0;
    cudaEventElapsedTime(&mem_transfer, transfer_start, transfer_stop);

    cout << "Transfer time: " << mem_transfer - milliseconds << " ms" << endl;

    cout << A[0] << " : Matrix A (CPU)" << endl;
    cout << B[0] << " : Matrix B (GPU)" << endl;

    compare_matrices(A[0], B[0]);

    cudaFree(d_B);
    cudaFree(d_Sum);

    delete[] A;
    delete[] B;
    delete[] cpuSum;
    delete[] gpuSum;
    
    return 0;
}

void init_matrix(int *A, int *B, int Width){
    int init = 1325;
    for(int i = 0; i < Width; i++){
        init = 3125 * init % 6553;
        A[i] = (init - 1000) % 97;
        B[i] = (init - 1000) % 97;
    }
}

void ParPrefix(int* x, int* y, int Width){
    y[0] = x[0];
    for(int i = 1; i < Width; i++){
        y[i] = y[i-1] + x[i];
    }
}

// Parallel Prefix Sum Kernel for CUDA
__global__ void ParPrefixKernel(int* x,int* y, int Width){
    __shared__ int scan_array[2 * blockDim.x];

    unsigned int threadID = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    scan_array[threadID] = x[start + threadID];
    scan_array[blockDim.x + threadID] = x[start + blockDim.x + threadID];

    // scan_array[threadID] = (i < Width) ? x[i] : 0;
    __syncthreads();

    // Reduction step pseudo code

    int stride = 1;
    int index;
    while(stride <= blockDim.x){
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < 2 * blockDim.x){
            scan_array[index] += scan_array[index - stride];
        }
        stride = stride * 2;

        __syncthreads();
    }

    //Post Scan step pseudo code

    stride = blockDim.x / 2;
    while(stride > 0){
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index + stride < 2 * blockDim.x){
            scan_array[index + stride] += scan_array[index];
        }
        stride = stride / 2;
        __syncthreads();
    }

    __synchthreads();

    x[start + threadID] = scan_array[threadID];
    x[start + blockDim.x + threadID] = scan_array[blockDim.x + threadID];
}

void compare_matrices(int cpu_result, int gpu_result){
    if(cpu_result != gpu_result){
        cout << "Sums are not equal" << endl;
        return;
    }
    cout << "Sums are equal" << endl;
}

void print_matrix(int *matrix, const char *name){
    cout << name << ":" << endl;
    for(int i = 0; i < 25; i++){
        cout << matrix[i] << ", ";
    }
    cout << endl;
}