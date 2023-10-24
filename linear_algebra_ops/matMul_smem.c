#include <stdio.h>
#include <CL/cl.h>

#define MATRIX_SIZE 4 // Adjust as needed
#define TILE_SIZE 2    // Adjust the tile size as needed

int main() {
    // Load OpenCL platform and create a context
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Load and compile the OpenCL kernel
    FILE* kernelFile = fopen("matMul_smem.cl", "r");
    if (!kernelFile) {
        printf("Failed to open kernel source file.\n");
        return 1;
    }
    char* source = (char*)malloc(10000);
    size_t sourceSize = fread(source, 1, 10000, kernelFile);
    fclose(kernelFile);

    
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, NULL);
    // clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_int buildStatus = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (buildStatus != CL_SUCCESS) {
    // Get the size of the build log
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

    // Allocate memory for the build log
    char* buildLog = (char*)malloc(logSize);

    // Get the build log
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);

    // Print the build log or handle the errors as needed
    printf("Build log:\n%s\n", buildLog);

    free(buildLog);
    }


    cl_kernel kernel = clCreateKernel(program, "matrixMultiplication", NULL);

    // Create input and output matrices, and allocate memory
    float* A = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* B = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* C = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* C_h = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    int matrixSize = MATRIX_SIZE;
    int tileSize = TILE_SIZE;
    
    // Initialize A and B matrices (you can replace this with your data)
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Multiply the above 2 matrices using C++
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            float sum = 0.0f;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
            }
            C_h[i * MATRIX_SIZE + j] = sum;
        }
    }


    // Create buffer objects for input and output matrices
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE* MATRIX_SIZE, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE* MATRIX_SIZE, B, 0, NULL, NULL);
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &matrixSize);
    clSetKernelArg(kernel, 4, sizeof(int), &tileSize);
    

    // Execute the kernel
    size_t globalWorkSize[2] = { MATRIX_SIZE, MATRIX_SIZE };
    size_t localWorkSize[2] = { TILE_SIZE, TILE_SIZE };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    
    // Read the result
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), C, 0, NULL, NULL);

    // Print the result
    printf("Matrix multiplication result:\n");
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        printf("%f ", C[i]);
    }
    printf("\n");
    // print C_h
    printf("Matrix multiplication result from host:\n");
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        printf("%f ", C_h[i]);
    }
    printf("\n");
    // Compare C with C_h
    int correct = 1;
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        if (C[i] != C_h[i]) {
            correct = 0;
        }
    }
    if (correct) {
        printf("The result is correct.\n");
    }
    else {
        printf("The result is incorrect.\n");
    }
    // Cleanup and release resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(A);
    free(B);
    free(C);
    free(source);

    return 0;
}
