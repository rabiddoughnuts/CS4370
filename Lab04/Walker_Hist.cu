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
void print_results(const char* label, unsigned int* histo, long size, long duration);
void check_results(unsigned int* cpu, unsigned int* gpu, unsigned int* gpu_shared);

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

    // Print array size and first 10 elements of input
    cout << "Array size: " << size << endl;
    cout << "First 10 elements of input array: ";
    for (int i = 0; i < 10 && i < size; i++) {
        cout << (int)input[i] << " ";
    }
    cout << endl;

    // CPU computation
    auto start = high_resolution_clock::now();
    cpuHist(input, cpu, size);
    auto end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(end - start).count();
    print_results("CPU", cpu, size, cpu_duration);

    // GPU computation with global memory
    cudaMemcpy(d_input, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, 256 * sizeof(unsigned int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Thread block size: " << threadsPerBlock << endl;
    cout << "Number of thread blocks: " << blocksPerGrid << endl;

    start = high_resolution_clock::now();
    hist_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histo, size);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    auto gpu_duration = duration_cast<milliseconds>(end - start).count();
    cudaMemcpy(gpu, d_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    print_results("GPU (global memory)", gpu, size, gpu_duration);

    // GPU computation with shared memory
    cudaMemset(d_histo, 0, 256 * sizeof(unsigned int));

    start = high_resolution_clock::now();
    histo_shared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histo, size);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    auto gpu_shared_duration = duration_cast<milliseconds>(end - start).count();
    cudaMemcpy(gpu_shared, d_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    print_results("GPU (shared memory)", gpu_shared, size, gpu_shared_duration);

    // Check results
    check_results(cpu, gpu, gpu_shared);

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

void print_results(const char* label, unsigned int* histo, long size, long duration) {
    cout << label << " computation time: " << duration << " ms" << endl;
    cout << "First 10 elements of " << label << " output: ";
    for (int i = 0; i < 10; i++) {
        cout << histo[i] << " ";
    }
    cout << endl;
}

void check_results(unsigned int* cpu, unsigned int* gpu, unsigned int* gpu_shared) {
    bool match = true;
    for (int i = 0; i < 256; i++) {
        if (cpu[i] != gpu[i]) {
            cout << "Mismatch between CPU and GPU (global memory) at index " << i << ": CPU = " << cpu[i] << ", GPU = " << gpu[i] << endl;
            match = false;
            break;
        }
    }
    if (match) {
        cout << "CPU and GPU (global memory) outputs match." << endl;
    }

    match = true;
    for (int i = 0; i < 256; i++) {
        if (cpu[i] != gpu_shared[i]) {
            cout << "Mismatch between CPU and GPU (shared memory) at index " << i << ": CPU = " << cpu[i] << ", GPU = " << gpu_shared[i] << endl;
            match = false;
            break;
        }
    }
    if (match) {
        cout << "CPU and GPU (shared memory) outputs match." << endl;
    }
}