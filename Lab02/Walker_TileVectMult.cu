// Brandon Walker
// CS4370
// Parallel Programming Many-Core GPUs
// Meilin Liu
// 9-Oct-2024
// Tiled Matrix Multiplication

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void init_matrix(int *A, int *B, int Width);
void mult_matrix_cpu(int* A, int* B, int* C, int Width);
__global__ void mult_matrix_gpu(int* d_A, int* d_B, int* d_C, int Width);
void compare_matrices(int *cpu_result, int *gpu_result, int Width);
void print_matrix(int *matrix, int Width, const char *name);

int main(){
    int Width, block_size;

    // Get Matrix size from user
    cout << "Enter size of the Width x Width matrix: ";
    cin >> Width;

    // Get bloack size from user
    cout << "Enter the block size for CUDA: ";
    cin >> block_size;

    // Allocate memory for matrices
    int *A = new int[Width * Width];
    int *B = new int[Width * Width];
    int *C_cpu = new int[Width * Width];
    int *C_gpu = new int[Width * Width];

    init_matrix(A, B, Width);

    print_matrix(A, Width, "Matrix A");
    print_matrix(B, Width, "Matrix B");

    auto start_cpu = chrono::high_resolution_clock::now();

    mult_matrix_cpu(A, B, C_cpu, Width);

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_cpu = end_cpu - start_cpu;
    cout << "CPU time: " << duration_cpu.count() << " ms" << endl;

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, Width * Width * sizeof(int));
    cudaMalloc(&d_B, Width * Width * sizeof(int));
    cudaMalloc(&d_C, Width * Width * sizeof(int));

    cudaMemcpy(d_A, A, Width * Width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Width * Width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((Width + block_size - 1) / block_size, (Width + block_size - 1) / block_size);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);
    mult_matrix_gpu<<<dimGrid, dimBlock, shared_mem_size>>>(d_A, d_B, d_C, Width);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    cout << "GPU time: " << milliseconds << " ms" << endl;

    cudaMemcpy(C_gpu, d_C, Width * Width * sizeof(int), cudaMemcpyDeviceToHost);

    print_matrix(C_gpu, Width, "Matrix C (GPU)");

    compare_matrices(C_cpu, C_gpu, Width);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;
    
    return 0;
}

void init_matrix(int *A, int *B, int Width){
    int init=1325;
    for(int row = 0; row < Width; row++){
        for(int col = 0; col < Width; col++){
            init= 3125 * init % 6553;
            A[row * Width + col] = (init - 1000) / 6553;
            B[row * Width + col] = init % 251;
        }
    }
}

/**
void init_matrix(int *A, int *B, int Width){
    int init=1325;
    for(int row = 0; row < Width; row++){
        for(int col = 0; col < Width; col++){
            init= (3125 * init) % 65536;
            A[row * Width + col] = (init - 32768) / 6553;
            B[row * Width + col] = init % 1000;
        }
    }
}
**/

void mult_matrix_cpu(int *A, int *B, int *C, int Width){
    for(int row = 0; row < Width; row++){
        for(int col = 0; col < Width; col++){
            int sum = 0;
            for(int k = 0; k < Width; k++){
                int m = A[row * Width + k];
                int n = B[k * Width + col];
                sum += m * n;
            }
            C[row * Width + col] = sum;
        }
    }
}

__global__ void mult_matrix_gpu(int* d_A, int* d_B, int* d_C, int Width){
    extern __shared__ int shared_mem[];
    int* ds_A = shared_mem;
    int* ds_B = shared_mem + blockDim.x * blockDim.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int P = 0;

    for (int k = 0; k < (Width + blockDim.x - 1) / blockDim.x; ++k) {
        if (row < Width && k * blockDim.x + tx < Width) {
            ds_A[ty * blockDim.x + tx] = d_A[row * Width + k * blockDim.x + tx];
        } else {
            ds_A[ty * blockDim.x + tx] = 0;
        }
        if (col < Width && k * blockDim.x + ty < Width) {
            ds_B[ty * blockDim.x + tx] = d_B[(k * blockDim.x + ty) * Width + col];
        } else {
            ds_B[ty * blockDim.x + tx] = 0;
        }

        __syncthreads();
        
        for (int n = 0; n < blockDim.x; ++n) {
            P += ds_A[ty * blockDim.x + n] * ds_B[n * blockDim.x + tx];
        }

        __syncthreads();
    }

    if (row < Width && col < Width) {
        d_C[row * Width + col] = P;
    }
}

void compare_matrices(int *cpu_result, int *gpu_result, int Width){
    for(int row = 0; row < Width; row++){
        for(int col = 0; col < Width; col++){
            int idx = row * Width + col;
            if(cpu_result[idx] != gpu_result[idx]){
                cout << "Mismatch at index (" << row << ", " << col << ")! CPU: " << cpu_result[idx] << ", GPU: " << gpu_result[idx] << endl;
                return;
            }
        }
        
    }
    cout << "CPU and GPU results match!" << endl;
}

void print_matrix(int *matrix, int Width, const char *name){
    cout << name << ":" << endl;
    if (Width <= 20){
        for(int row = 0; row < Width; row++){
            for(int col = 0; col < Width; col++){
                cout << matrix[row * Width + col] << " ";
            }
            cout << endl;
        }
    } else {
        cout << "Matrix too large, printing only the first row:" << endl;
        for(int col = 0; col < Width; col++){
            cout << matrix[col] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
