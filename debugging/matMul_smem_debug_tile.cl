__kernel void matrixMultiplication(
    __global float* A, 
    __global float* B, 
    __global float* C,
    int matrixSize,
    int tileSize,
    __global float* tileA,
    __global float* tileB
) {
    __local float localA[2][2];
    __local float localB[2][2];

    int row = get_local_id(0);
    int col = get_local_id(1);
    int globalRow = get_group_id(0) * tileSize + row;
    int globalCol = get_group_id(1) * tileSize + col;
    float sum = 0.0f;

    tileA[globalRow*tileSize+globalCol] = A[globalRow * matrixSize + globalCol];
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int tile = 0; tile < matrixSize / tileSize; tile++) {
        // Load the tiles from global memory into local memory
        localA[row][col] = A[globalRow * matrixSize + (tile * tileSize + col)];
        localB[row][col] = B[(tile * tileSize + row) * matrixSize + globalCol];
        if (get_group_id(0)==0 && get_group_id(1) == 0 && tile == 1) {
            tileA[row*tileSize+col] = A[globalRow * matrixSize + (tile * tileSize + col)];
            tileB[row*tileSize+col] = B[(tile * tileSize + row) * matrixSize + globalCol];
        }
        // Wait for all threads to finish loading their tiles
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two tiles together
        for (int i = 0; i < tileSize; i++) {
            sum += localA[row][i] * localB[i][col];
        }

        // Wait for all threads to finish multiplying the tiles
        barrier(CLK_LOCAL_MEM_FENCE);
    };
    
    // Use a global barrier to ensure all work-items in the work-group have completed their computation
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Write the result to global memory
    C[globalRow * matrixSize + globalCol] = sum;
}
