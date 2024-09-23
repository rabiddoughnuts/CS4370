// This will be a program to do matrix addition using CUDA and on the CPU and it will compare the results.

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void init_matrix(int *A, int *B, int N);
void add_matrix_cpu(int *A, int *B, int *C, int N);
__global__ void add_matrix_gpu(int *A, int *B, int *C, int N);
void compare_matrices(int *cpu_result, int *gpu_result, int N);
void print_matrix(int *matrix, int N, const char *name);

int main(){
    int N, block_size;

    // Get Matrix size from user
    cout << "Enter size of the N x N matrix: ";
    cin >> N;

    // Get bloack size from user
    cout << "Enter the block size for CUDA: ";
    cin >> block_size;

    // Allocate memory for matrices
    int *A = new int[N * N];
    int *B = new int[N * N];
    int *C_cpu = new int[N * N];
    int *C_gpu = new int[N * N];

    init_matrix(A, B, N);

    print_matrix(A, N, "Matrix A");
    print_matrix(B, N, "Matrix B");

    auto start_cpu = chrono::high_resolution_clock::now();

    add_matrix_cpu(A, B, C_cpu, N);

    auto ent_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_cpu = end_cpu - start_cpu;
    cout << "CPU time: " << duration_cpu.count() << " ms" << endl;

    print_matrix(C_cpu, N, "Matrix C (CPU)");

    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    add_matrix_gpu<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronise(stop_gpu);

    float milliseconds = 0;
    cudaEventElapsedTimed(milliseconds, start_gpu, stop_gpu);

    cout << "GPU time: " << milliseconds << " ms" << endl;

    cudaMemcpy(C_gpu, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    print_matrix(C_gpu, N, "Matrix C (GPU)");

    compare_matrices(C_cpu, C_gpu, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete(A);
    delete(B);
    delete(C_cpu);
    delete(C_gpu);
    
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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * N + col;

    if( row < N && col < N){
        C[index] = A[index] + B[index];
    }
}

void compare_matrices(int *cpu_result, int *gpu_result, int N){
    for(int i = 0; i < N; i++){
        if(cpu_result[i] != gpu_result[i]){
            cout << "Mismatch at index " << i << "! CPU: " << cpu_result[i] << ", GPU: " << gpu_result[i] << endl;
            return;
        }
    }
    cout << "CPU and GPU results match!" << endl;
}

void print_matrix(int *matrix, int N, const char *name){
    cout << name << ":" << endl;
    if (N <= 20){
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                cout << matrix[i * N + j] << " ";
            }
            cout << endl;
        }
    } else {
        cout << "Matrix too large, printing only the first row:" << endl;
        for(int j = 0; j < N; j++){
            cout << matrix[j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}