// Brandon Walker
// CS4370
// Parallel Programming Many-Core GPUs
// Meilin Liu
// 18-Nov-2024
// Histogram

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;



int main(){
    long size;

    cout << "Enter size of the Array: ";
    cin >> size;

    unsigned char* input[size];
    unsigned int cpu[256];
    unsigned int gpu[256];

}

void init_array(int* cpu, int* gpu, int size){
    int init = 1325;
    for(int i = 0; i < size; i++){
        init = 3125 * init % 65537;
        cpu[i] = init % 256;
        gpu[i] = init % 256;
    }
}

void cpuHist(int* A, int* histo, int size){
    for(int i = 0; i < 256; i++){
        histo[i] = 0;
    }
    for (int i = 0; i < size; i++){
        histo[A[i]]++;
    }
}

__global__ void hist_kernel(unsigned char* A, int* histo, long size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < size){
        atomicAdd(&histo[A[i]], 1);
        i+= stride;
    }
}

__global__ void histo_shared(unsigned char* A, int* histo, long size){
    __shared__ unsigned int histo_private[256];
    
    if(threadIdx.x < 256){
        histo_private[threadIdx.x] = 0;
    }
    __syncthreads();
    
}