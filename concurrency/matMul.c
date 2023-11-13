#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MATRIX_SIZE 800

double get_event_exec_time(cl_event event)
{
    cl_ulong start_time, end_time;
    /*Get start device counter for the event*/
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong),
                            &start_time,
                            NULL);
    /*Get end device counter for the event*/
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong),
                            &end_time,
                            NULL);
    /*Convert the counter values to milli seconds*/
    double total_time = (double)(end_time - start_time);
    return total_time;
}
// Function to read the content of a file into a string
const char *readKernelSourceFromFile(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        return NULL; // File not found or couldn't be opened
    }

    // Find the size of the file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    // Allocate memory for the file content plus a null-terminating character
    char *source_code = (char *)malloc(file_size + 1);
    if (source_code == NULL)
    {
        fclose(file);
        return NULL; // Memory allocation failed
    }

    // Read the file content
    size_t read_size = fread(source_code, 1, file_size, file);
    fclose(file);

    if (read_size != file_size)
    {
        free(source_code);
        return NULL; // Reading the file failed
    }

    // Null-terminate the string
    source_code[file_size] = '\0';

    return source_code;
}

int main()
{
    // Specify the file name containing the kernel code
    int matrix_size = MATRIX_SIZE;
    const char *kernel_file = "concurrency/matMul.cl";

    // Read the kernel code from the file
    const char *kernelSource = readKernelSourceFromFile(kernel_file);

    if (kernelSource == NULL)
    {
        printf("Failed to read the kernel source from the file.\n");
        return 1;
    }

    float a[MATRIX_SIZE * MATRIX_SIZE], b[MATRIX_SIZE * MATRIX_SIZE];

    // Initialize input vectors
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    // Output vector
    float result[MATRIX_SIZE * MATRIX_SIZE];

    // Load the OpenCL platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Load the OpenCL device (GPU in this case)
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Create and compile the OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matMul", NULL);

    // Create OpenCL buffers for vectors
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);

    // Write data to the OpenCL buffers
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, b, 0, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(int), &matrix_size);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferResult);

    // Enqueue the OpenCL kernel for execution
    size_t globalWorkSize[2] = {MATRIX_SIZE, MATRIX_SIZE};
    cl_event kernel_event1, kernel_event2;
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &kernel_event1);
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &kernel_event2);
    clFinish(queue);
    double kernel1_time = get_event_exec_time(kernel_event1);
    printf("kernel1 time: %f\n", kernel1_time);
    double kernel2_time = get_event_exec_time(kernel_event2);
    printf("kernel2 time: %f\n", kernel2_time);

    // Read the result from the OpenCL buffer
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, result, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // // Print the result
    // printf("Matrix multiplication result:\n");
    // for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++)
    // {
    //     printf("%f ", result[i]);
    // }
    // printf("\n");
    // free(kernelSource);

    return 0;
}
