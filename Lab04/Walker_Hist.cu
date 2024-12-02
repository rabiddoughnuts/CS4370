// Brandon Walker
// CS4370
// Parallel Programming Many-Core GPUs
// Meilin Liu
// 30-Nov-2024
// Histogram

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

void init_array(unsigned char* input, long size);
void cpuHist(unsigned char* A, unsigned int* histo, long size);
__global__ void hist_kernel(unsigned char* A, unsigned int* histo, long size);
__global__ void histo_shared(unsigned char* A, unsigned int* histo, long size);

int main(){
    long size;

    cout << "Enter size of the Array: ";
    cin >> size;

    unsigned char* input = new unsigned char[size];
    unsigned int cpu[256] = {0};
    unsigned int gpu[256] = {0};
    unsigned int gpu_shared[256] = {0};

    unsigned char* d_input;
    unsigned int* d_histo;

    cudaMalloc(&d_input, size * sizeof(unsigned char));
    cudaMalloc(&d_histo, 256 * sizeof(unsigned int));

    init_array(input, size);

    // CPU computation
    auto start = high_resolution_clock::now();
    cpuHist(input, cpu, size);
    auto end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(end - start).count();
    cout << "CPU computation time: " << cpu_duration << " ms" << endl;

    // GPU computation with global memory
    cudaMemcpy(d_input, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, 256 * sizeof(unsigned int));

    start = high_resolution_clock::now();
    hist_kernel<<<256, 256>>>(d_input, d_histo, size);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    auto gpu_duration = duration_cast<milliseconds>(end - start).count();
    cout << "GPU computation time (global memory): " << gpu_duration << " ms" << endl;

    cudaMemcpy(gpu, d_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // GPU computation with shared memory
    cudaMemset(d_histo, 0, 256 * sizeof(unsigned int));

    start = high_resolution_clock::now();
    histo_shared<<<256, 256>>>(d_input, d_histo, size);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    auto gpu_shared_duration = duration_cast<milliseconds>(end - start).count();
    cout << "GPU computation time (shared memory): " << gpu_shared_duration << " ms" << endl;

    cudaMemcpy(gpu_shared, d_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_histo);
    delete[] input;

    return 0;
}

void init_array(unsigned char* input, long size){
    int init = 1325;
    for(long i = 0; i < size; i++){
        init = 3125 * init % 65537;
        input[i] = init % 256;
    }
}

void cpuHist(unsigned char* A, unsigned int* histo, long size){
    for(int i = 0; i < 256; i++){
        histo[i] = 0;
    }
    for (long i = 0; i < size; i++){
        histo[A[i]]++;
    }
}

__global__ void hist_kernel(unsigned char* A, unsigned int* histo, long size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < size){
        atomicAdd(&histo[A[i]], 1);
        i += stride;
    }
}

__global__ void histo_shared(unsigned char* A, unsigned int* histo, long size){
    __shared__ unsigned int histo_private[256];
    
    if(threadIdx.x < 256){
        histo_private[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < size){
        atomicAdd(&histo_private[A[i]], 1);
        i += stride;
    }

    __syncthreads();
    if(threadIdx.x < 256){
        atomicAdd(&histo[threadIdx.x], histo_private[threadIdx.x]);
    }
}