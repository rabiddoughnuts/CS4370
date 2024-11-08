// Brandon Walker
// CS4370
// Parallel Programming Many-Core GPUs
// Meilin Liu
// 8-Nov-2024
// Sum Reduction

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void init_matrix(int *A, int *B, int Width);
void SumReduction(int* x, int Width);
__global__ void SumReductionKernel(int* x, int Width);
void compare_matrices(int* cpu_result, int* gpu_result, int Width);
void print_matrix(int *matrix, int Width, const char *name);

int main(){
    int Width, block_size;

    // Get Matrix size from user
    cout << "Enter size of the Array: ";
    cin >> Width;

    // Get block size from user
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

    int *d_B;
    cudaMalloc(&d_B, Width * sizeof(int));

    cudaEvent_t transfer_start, transfer_stop;
    cudaEventCreate(&transfer_start);
    cudaEventCreate(&transfer_stop);

    cudaEventRecord(transfer_start);
    cudaMemcpy(d_B, B, Width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(block_size);
    dim3 dimGrid((Width + block_size - 1) / block_size);
    // dim3 dimGrid((Width + 2 * block_size - 1) / (2 * block_size));
    int numBlocks = dimGrid.x;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    size_t shared_mem_size = block_size * sizeof(int);

    while(dimGrid.x > 1){
        SumReductionKernel<<<dimGrid, dimBlock, shared_mem_size>>>(d_B, Width);
        cudaDeviceSynchronize();
        Width = dimGrid.x;
        dimGrid.x = (Width + dimBlock.x - 1) / dimBlock.x;
    }

    SumReductionKernel<<<1, dimBlock, shared_mem_size>>>(d_B, Width);
    // cudaDeviceSynchronize();

    //int num_blocks = dimGrid.x;
    // while (num_blocks > 1) {
    //     dim3 new_dimGrid((num_blocks + 2 * block_size - 1) / (2 * block_size));
    //     SumReductionKernel<<<new_dimGrid, dimBlock, shared_mem_size>>>(d_B, num_blocks);
    //     cudaDeviceSynchronize();
    //     num_blocks = new_dimGrid.x;
    // }

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cudaMemcpy(B, d_B, Width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(transfer_stop);
    cudaEventSynchronize(transfer_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);


    print_matrix(A, Width, "Output array from CPU");
    print_matrix(B, Width, "Output array from GPU");

    cout << "Array size: " << Width << endl;
    cout << "Block size: " << block_size << endl;
    cout << "Number of thread blocks initiated: " << numBlocks << endl;

    cout << "CPU time: " << duration_cpu.count() << " ms" << endl;
    cout << "GPU time: " << milliseconds << " ms" << endl;

    float mem_transfer = 0;
    cudaEventElapsedTime(&mem_transfer, transfer_start, transfer_stop);

    cout << "Transfer time: " << mem_transfer - milliseconds << " ms" << endl;

    compare_matrices(A, B, Width);

    cudaFree(d_B);

    delete[] A;
    delete[] B;
    
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
    // if (i + blockDim.x < Width){
    //     sdata[threadID] += x[i + blockDim.x];
    // }
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