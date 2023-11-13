#include <CL/cl.h>
#include <stdio.h>

#define MAX_SOURCE_SIZE (0x100000)

const char *kernelSource = "__kernel void example_kernel(__global float *input, __global float *output) {\n"
                          "   int i = get_global_id(0);\n"
                          "   output[i] = input[i] * input[i];\n"
                          "}\n";

int main() {
    // Create OpenCL variables
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem inputBuffer, outputBuffer;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_event event;
    cl_uint num_platforms;
    cl_int ret;

    // Data
    const int dataSize = 10;
    float inputData[dataSize];
    float outputData[dataSize];

    // Initialize input data
    for (int i = 0; i < dataSize; ++i) {
        inputData[i] = i + 1;
    }

    // Get platform and device information
    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create command queue
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    // Create memory buffers on the device
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize * sizeof(float), NULL, &ret);
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize * sizeof(float), NULL, &ret);

    // Write data from host to device
    ret = clEnqueueWriteBuffer(command_queue, inputBuffer, CL_TRUE, 0, dataSize * sizeof(float), inputData, 0, NULL, NULL);

    // Create the kernel program
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "example_kernel", &ret);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);

    // Execute the OpenCL kernel
    size_t globalItemSize = dataSize;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalItemSize, NULL, 0, NULL, &event);

    // Wait for the command queue to finish
    clFinish(command_queue);

    // Get kernel execution time
    cl_ulong time_start, time_end;
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double executionTime = (double)(time_end - time_start) * 1.0e-6;

    // Read output buffer from the device to the host
    ret = clEnqueueReadBuffer(command_queue, outputBuffer, CL_TRUE, 0, dataSize * sizeof(float), outputData, 0, NULL, NULL);

    // Display the results and timing
    printf("Output:\n");
    for (int i = 0; i < dataSize; ++i) {
        printf("%0.2f ", outputData[i]);
    }
    printf("\nKernel Execution Time: %0.3f ms\n", executionTime);

    // Cleanup
    ret = clFlush(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(inputBuffer);
    ret = clReleaseMemObject(outputBuffer);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}
