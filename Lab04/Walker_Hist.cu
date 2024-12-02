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
void print_array(const char* label, unsigned int* histo);
void print_timing(const char* label, float duration);
void print_memory_transfer_timing(float total, float to_device, float from_device);
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
    auto start_cpu = high_resolution_clock::now();
    cpuHist(input, cpu, size);
    auto end_cpu = high_resolution_clock::now();
    duration<float, milli> duration_cpu = end_cpu - start_cpu;

    // GPU computation with global memory
    auto start_transfer_to_gpu = high_resolution_clock::now();
    cudaMemcpy(d_input, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, 256 * sizeof(unsigned int));
    auto end_transfer_to_gpu = high_resolution_clock::now();
    duration<float, milli> duration_transfer_to_gpu = end_transfer_to_gpu - start_transfer_to_gpu;

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    hist_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histo, size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);

    auto start_transfer_from_gpu = high_resolution_clock::now();
    cudaMemcpy(gpu, d_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    auto end_transfer_from_gpu = high_resolution_clock::now();
    duration<float, milli> duration_transfer_from_gpu = end_transfer_from_gpu - start_transfer_from_gpu;

    float total_gpu_duration = gpu_duration + duration_transfer_to_gpu.count() + duration_transfer_from_gpu.count();

    // GPU computation with shared memory
    cudaMemset(d_histo, 0, 256 * sizeof(unsigned int));

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    histo_shared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histo, size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_shared_duration = 0;
    cudaEventElapsedTime(&gpu_shared_duration, start_gpu, stop_gpu);

    start_transfer_from_gpu = high_resolution_clock::now();
    cudaMemcpy(gpu_shared, d_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    end_transfer_from_gpu = high_resolution_clock::now();
    duration_transfer_from_gpu = end_transfer_from_gpu - start_transfer_from_gpu;

    float total_gpu_shared_duration = gpu_shared_duration + duration_transfer_to_gpu.count() + duration_transfer_from_gpu.count();

    // Print results
    cout << "First 10 elements of CPU output: ";
    print_array("CPU", cpu);

    cout << "First 10 elements of GPU output (global memory): ";
    print_array("GPU", gpu);

    cout << "First 10 elements of GPU output (shared memory): ";
    print_array("GPU (shared memory)", gpu_shared);

    print_timing("CPU computation time", duration_cpu.count());
    print_timing("GPU computation time (global memory)", gpu_duration);
    print_timing("GPU computation time (shared memory)", gpu_shared_duration);

    print_memory_transfer_timing(duration_transfer_to_gpu.count() + duration_transfer_from_gpu.count(), duration_transfer_to_gpu.count(), duration_transfer_from_gpu.count());

    print_timing("Total GPU time (global memory)", total_gpu_duration);
    print_timing("Total GPU time (shared memory)", total_gpu_shared_duration);

    cout << "Array size: " << size << endl;
    cout << "Thread block size: " << threadsPerBlock << endl;
    cout << "Number of thread blocks: " << blocksPerGrid << endl;

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

void print_array(const char* label, unsigned int* histo) {
    for (int i = 0; i < 10; i++) {
        cout << histo[i] << " ";
    }
    cout << endl;
}

void print_timing(const char* label, float duration) {
    cout << label << ": " << duration << " ms" << endl;
}

void print_memory_transfer_timing(float total, float to_device, float from_device) {
    cout << "Memory transfer time (total, to, from): " << total << " ms, " << to_device << " ms, " << from_device << " ms" << endl;
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