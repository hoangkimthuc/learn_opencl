__kernel void matrixMultiplication(
    __global float* A, 
    __global float* B, 
    __global float* C,
    __local float* localA,
    __local float* localB,
    int matrixSize,
    int tileSize
) {
    int row = get_local_id(0);
    int col = get_local_id(1);
    int globalRow = get_group_id(0) * tileSize + row;
    int globalCol = get_group_id(1) * tileSize + col;

    float sum = 0.0f;
    for (int tile = 0; tile < matrixSize / tileSize; tile++) {
        // Load the tiles from global memory into local memory
        localA[row*tileSize + col] = A[globalRow * matrixSize + (tile * tileSize + col)];
        localB[row*tileSize + col] = B[(tile * tileSize + row) * matrixSize + globalCol];

        // Wait for all threads to finish loading their tiles
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two tiles together
        for (int i = 0; i < tileSize; i++) {
            sum += localA[row*tileSize + i] * localB[i*tileSize + col];
        }

        // Wait for all threads to finish multiplying the tiles
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result to global memory
    C[globalRow * matrixSize + globalCol] = sum;
    
}
