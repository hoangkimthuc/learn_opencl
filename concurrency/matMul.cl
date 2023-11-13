kernel void matMul(global const float* a, global const float* b, int matrix_size, global float* result) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for (int i = 0; i < matrix_size; i++){
        sum += a[row * matrix_size + i] * b[i * matrix_size + col];
    }
    result[row * matrix_size + col] = sum;
};

