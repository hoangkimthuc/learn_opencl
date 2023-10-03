#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define VECTOR_SIZE 10

// OpenCL kernel to add two vectors
const char* kernelSource =
    "__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* result) {\n"
    "    int i = get_global_id(0);\n"
    "    result[i] = a[i] + b[i];\n"
    "}\n";

int main() {
    // Input vectors
    float a[VECTOR_SIZE], b[VECTOR_SIZE];

    // Initialize input vectors
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Output vector
    float result[VECTOR_SIZE];

    // Load the OpenCL platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Load the OpenCL device (GPU in this case)
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create and compile the OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", NULL);

    // Create OpenCL buffers for vectors
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * VECTOR_SIZE, NULL, NULL);

    // Write data to the OpenCL buffers
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, b, 0, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);

    // Enqueue the OpenCL kernel for execution
    size_t globalWorkSize = VECTOR_SIZE;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

    // Wait for the kernel to finish
    clFinish(queue);

    // Read the result from the OpenCL buffer
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, result, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Print the result
    printf("Vector addition result:\n");
    for (int i = 0; i < VECTOR_SIZE; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    return 0;
}
