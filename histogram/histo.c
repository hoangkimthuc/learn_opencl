#include <CL/cl.h>
#include <stdio.h>

#define HIST_BINS 4
#define NUM_DATA 8  // Adjust as needed

int main() {
    // Create OpenCL context
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Load kernel source code (assuming it's in a file named "histogram.cl")
    FILE* file = fopen("histo.cl", "r");
    if (!file) {
        perror("Failed to open kernel source file");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);
    char* kernelSource = (char*)malloc(fileSize + 1);
    fread(kernelSource, 1, fileSize, file);
    kernelSource[fileSize] = '\0';
    fclose(file);

    // Create program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "histogram", NULL);

    // Create and populate input data array
    int numData = NUM_DATA;
    int data[NUM_DATA];
    // Fill data with values...
    for (int i = 0; i < NUM_DATA; i++) {
        data[i] = rand() % HIST_BINS;
        printf("%d ", data[i]);
    }
    printf("\n");

    // Create OpenCL buffers for data and histogram
    cl_mem dataBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NUM_DATA, data, NULL);
    cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * HIST_BINS, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dataBuffer);
    clSetKernelArg(kernel, 1, sizeof(int), &numData);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &histogramBuffer);

    // Enqueue the kernel for execution
    size_t globalWorkSize = NUM_DATA;
    size_t localWorkSize = 2;  // Adjust as needed
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

    // Read the results from the histogram buffer
    int histogram[HIST_BINS];
    clEnqueueReadBuffer(queue, histogramBuffer, CL_TRUE, 0, sizeof(int) * HIST_BINS, histogram, 0, NULL, NULL);

    // print result
    for (int i = 0; i < HIST_BINS; i++) {
        printf("%d ", histogram[i]);
        printf("\n");
    }

    // Cleanup
    clReleaseMemObject(dataBuffer);
    clReleaseMemObject(histogramBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Process the histogram data as needed...

    return 0;
}
