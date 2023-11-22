#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

#define MATRIX_SIZE 512

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
    const char *kernel_file = "matMul.cl";

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
        a[i] = 1;
        b[i] = 2;
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
    cl_command_queue queue1 = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, NULL);
    cl_command_queue queue2 = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, NULL);

    // Create and compile the OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel1 = clCreateKernel(program, "matMul", NULL);
    cl_kernel kernel2 = clCreateKernel(program, "matMul", NULL);

    // Start timing the kernel
    double start, end;
    double cpu_time_used;
    start = clock();
    // Create OpenCL buffers for vectors
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);

    // Write data to the OpenCL buffers
    clEnqueueWriteBuffer(queue1, bufferA, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue1, bufferB, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, b, 0, NULL, NULL);

    // Write data to the OpenCL buffers
    clEnqueueWriteBuffer(queue2, bufferA, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue2, bufferB, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, b, 0, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel1, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel1, 2, sizeof(int), &matrix_size);
    clSetKernelArg(kernel1, 3, sizeof(cl_mem), &bufferResult);
    
    
    // Enqueue the OpenCL kernel for execution
    size_t globalWorkSize[2] = {MATRIX_SIZE, MATRIX_SIZE};
    cl_event event1, event2;
    clEnqueueNDRangeKernel(queue1, kernel1, 2, NULL, globalWorkSize, NULL, 0, NULL, &event1);
    clFinish(queue1);
    
    // Kernel 2
    cl_mem bufferA1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferB1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferResult1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    clEnqueueWriteBuffer(queue2, bufferA1, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue2, bufferB1, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, b, 0, NULL, NULL);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &bufferA1);
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &bufferB1);
    clSetKernelArg(kernel2, 2, sizeof(int), &matrix_size);
    clSetKernelArg(kernel2, 3, sizeof(cl_mem), &bufferResult1);
    clEnqueueNDRangeKernel(queue2, kernel2, 2, NULL, globalWorkSize, NULL, 0, NULL, &event2);
    clFinish(queue2);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for kernel execution: %f\n", cpu_time_used);

    // Read the result from the OpenCL buffer
    clEnqueueReadBuffer(queue1, bufferResult, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, result, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue1);
    clReleaseContext(context);
    // cpu kernel 
    float cpu_result[MATRIX_SIZE * MATRIX_SIZE];
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) { 
            float sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += a[i * MATRIX_SIZE + k] * b[k * MATRIX_SIZE + j];
            }
            cpu_result[i * MATRIX_SIZE + j] = sum;
        }
    }   
    // // verify the result
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        if (cpu_result[i] != result[i]) {
            printf("The result is wrong!\n");
            return 1;
        }
    }
    return 0;
}
