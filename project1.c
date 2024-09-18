// This will be a program to do matrix addition using CUDA and on the CPU and it will compare the results.


void main(){
    matrix A = init_matrix;
    matrix B = init_matrix;
    dim3 dimBlock(blocksize, blocksize, 1);
    dim3 dimGrid(ceiling (double (N)/dimBlock.x), ceiling (double (N)/dimBlock.y), 1);
    add_matrix_gpu<<<dimGrid, dimBlock>>>(a,b,c,N);
}

void init_matrix(){
    int *a, *b, *c;
    A=malloc(sizeof(int) * N * N); //N is the size
    //then malloc for b and x
    int init=1325;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            init=3125*init%65536;
            A[i][j] = (init - 32768) / 6553;
            B[i][j] = init % 1000;
        }
    }
}

void add_matrix_cpu(int *a, int *b, int *c, int N){
    int i, j, index;
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            index = i * N + j;
            c[index] = a[index] + b[index];
        }
    }
}