// Brandon Walker
// CS4370
// Parallel Programming Many-Core GPUs
// Meilin Liu
// 8-Nov-2024
// Parallel Prefix Sum

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void init_matrix(int *A, int *B, int Width);
void ParPrefix(int* x, int* y, int Width);
__global__ void ParPrefixKernel(int* x,int* y, int* sum, int Width);
__global__ void AddScannedBlockSums(int* x, int* y, int* sum, int Width);
void compare_matrices(int* cpu_result, int* gpu_result, int Width);
void print_matrix(int* matrix, int Width, const char *name);

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

    print_matrix(A, Width, "Matrix A");
    print_matrix(B, Width, "Matrix B");

    auto start_cpu = chrono::high_resolution_clock::now();

    ParPrefix(A, cpuSum, Width);

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_cpu = end_cpu - start_cpu;

    int* d_B;
    int* d_Sum;
    int* d_blockSums;
    cudaMalloc(&d_B, Width * sizeof(int));
    cudaMalloc(&d_Sum, Width * sizeof(int));
    cudaMalloc(&d_blockSums, ((Width + 2 * block_size - 1) / (2 * block_size)) * sizeof(int));

    cudaEvent_t transfer_start, transfer_stop;
    cudaEventCreate(&transfer_start);
    cudaEventCreate(&transfer_stop);

    cudaEventRecord(transfer_start);
    cudaMemcpy(d_B, B, Width * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Sum, gpuSum, Width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(block_size);
    dim3 dimGrid((Width + 2 * block_size - 1) / (2 * block_size));
    int num_blocks = dimGrid.x;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    size_t shared_mem_size = 2 * block_size * sizeof(int);

    ParPrefixKernel<<<dimGrid, dimBlock, shared_mem_size>>>(d_B, d_Sum, d_blockSums, Width);

        // Perform scan on block sums
    if (dimGrid.x > 1) {
        int* d_blockSumsScan;
        cudaMalloc(&d_blockSumsScan, dimGrid.x * sizeof(int));
        ParPrefixKernel<<<1, dimBlock, shared_mem_size>>>(d_blockSums, d_blockSumsScan, nullptr, dimGrid.x);
        cudaMemcpy(d_blockSums, d_blockSumsScan, dimGrid.x * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(d_blockSumsScan);
    }

    AddScannedBlockSums<<<dimGrid, dimBlock>>>(d_B, d_Sum, d_blockSums, Width);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    //cudaMemcpy(B, d_B, Width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuSum, d_Sum, Width * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(transfer_stop);
    cudaEventSynchronize(transfer_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    print_matrix(cpuSum, Width, "CPU Sum");
    print_matrix(gpuSum, Width, "GPU Sum");

    cout << "Array size: " << Width << endl;
    cout << "Thread block size: " << block_size << endl;
    cout << "Number of thread blocks initiated: " << num_blocks << endl;

    cout << "CPU time: " << duration_cpu.count() << " ms" << endl;
    cout << "GPU time: " << milliseconds << " ms" << endl;

    float mem_transfer = 0;
    cudaEventElapsedTime(&mem_transfer, transfer_start, transfer_stop);

    cout << "Transfer time: " << mem_transfer - milliseconds << " ms" << endl;

    // cout << A[0] << " : Matrix A (CPU)" << endl;
    // cout << B[0] << " : Matrix B (GPU)" << endl;

    compare_matrices(A[], B[], Width);

    cudaFree(d_B);
    cudaFree(d_Sum);
    cudaFree(d_blockSums);

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
__global__ void ParPrefixKernel(int* x, int* y, int* sum, int Width){
   extern __shared__ int scan_array[];

    unsigned int threadID = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    if (start + threadID < Width)
        scan_array[threadID] = x[start + threadID];
    else
        scan_array[threadID] = 0;
    if (start + blockDim.x + threadID < Width)
        scan_array[blockDim.x + threadID] = x[start + blockDim.x + threadID];
    else
        scan_array[blockDim.x + threadID] = 0;

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

    __syncthreads();

    if (start + threadID < Width)
        y[start + threadID] = scan_array[threadID];
    if (start + blockDim.x + threadID < Width)
        y[start + blockDim.x + threadID] = scan_array[blockDim.x + threadID];

    if (sum != nullptr && threadID == 0)
        sum[blockIdx.x] = scan_array[2 * blockDim.x - 1];
}

// Kernel to add scanned block sums to each element
__global__ void AddScannedBlockSums(int* x, int* y, int* sum, int Width) {
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    unsigned int threadID = threadIdx.x;
    if (blockIdx.x > 0) {
        if (start + threadID < Width)
            y[start + threadID] += sum[blockIdx.x - 1];
        if (start + blockDim.x + threadID < Width)
            y[start + blockDim.x + threadID] += sum[blockIdx.x - 1];
    }
}

void compare_matrices(int* cpu_result, int* gpu_result, int Width){
    for (int i = 0; i < Width; i++) {
        if(cpu_result[i] != gpu_result[i]){
            cout << "Sums are not equal at index " << i << endl;
            return;
        }
    }
    cout << "Sums are equal" << endl;
}

void print_matrix(int *matrix, int Width, const char *name){
    cout << name << ":" << endl;
    for(int i = 0; i < min(20, Width); i++){
        cout << matrix[i] << " ";
    }
    cout << endl;
}