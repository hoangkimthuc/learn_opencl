const char* kernelSource =
    "__kernel void matMul(__global const float* a, __global const float* b, __global float* result) {\n"
    "    int row = get_global_id(0);\n"
    "    int col = get_global_id(1);\n"
    "    float sum = 0.0f;\n"
    "    for (int i = 0; i < MATRIX_SIZE; i++){\n"
    "        sum += a[row * MATRIX_SIZE + i] * b[i * MATRIX_SIZE + col];\n"
    "    }\n"
    "    result[row * MATRIX_SIZE + col] = sum;\n"
    "}\n";
