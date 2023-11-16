#include <CL/cl.h>
#include <stdio.h>

#define ARRAY_SIZE 10

// Kernel source code
const char* kernelSource =
    "__kernel void simple_kernel(__global int* input, __global int* output) {\n"
    "    int gid = get_global_id(0);\n"
    "    printf(\"Global ID: %d, Input Value: %d\\n\", gid, input[gid]);\n"
    "    output[gid] = input[gid] * 2;\n"
    "}\n";

// Function to check OpenCL errors
void checkOpenCLError(cl_int error, const char* message) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "%s (Error code: %d)\n", message, error);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Get OpenCL platform and device
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    checkOpenCLError(context == NULL, "Failed to create OpenCL context");

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    checkOpenCLError(queue == NULL, "Failed to create command queue");

    // Create the program from the kernel source code
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    checkOpenCLError(program == NULL, "Failed to create program");

    // Build the program
    cl_int error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        // Print build log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        char* log = (char*)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

        fprintf(stderr, "Failed to build program. Build log:\n%s\n", log);
        free(log);

        exit(EXIT_FAILURE);
    }

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "simple_kernel", &error);
    checkOpenCLError(kernel == NULL, "Failed to create kernel");

    // Create input and output buffers
    int inputData[ARRAY_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int outputData[ARRAY_SIZE];

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(int) * ARRAY_SIZE, inputData, &error);
    checkOpenCLError(inputBuffer == NULL, "Failed to create input buffer");

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(int) * ARRAY_SIZE, NULL, &error);
    checkOpenCLError(outputBuffer == NULL, "Failed to create output buffer");

    // Set kernel arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    checkOpenCLError(error, "Failed to set kernel arguments");

    // Execute the kernel
    size_t globalWorkSize[1] = {ARRAY_SIZE};
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    checkOpenCLError(error, "Failed to enqueue kernel");

    // Read the result from the output buffer
    error = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0,
                                sizeof(int) * ARRAY_SIZE, outputData, 0, NULL, NULL);
    checkOpenCLError(error, "Failed to read output buffer");

    // Display the output
    printf("Output: ");
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        printf("%d ", outputData[i]);
    }
    printf("\n");

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
