#include <CL/cl.h>
#include <stdio.h>

int main() {
    // Define image and filter dimensions.
    #define IMAGEWIDTH 4
    #define IMAGEHEIGHT 4
    #define FILTERWIDTH 2
    #define FILTERHEIGHT 2

    int imageWidth = 4;
    int imageHeight = 4;
    int filterWidth = 2;
    int filterHeight = 2;
    int outputWidth = imageWidth - filterWidth + 1;
    int outputHeight = imageHeight - filterHeight + 1; 

    // Initialize image and filter data (you can read these from your file).
    int image[IMAGEWIDTH*IMAGEHEIGHT] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} ;  // Initialize your image data.
    int filter[FILTERWIDTH*FILTERHEIGHT] = {1, 1, 1, -1};  // Initialize your filter data.
    int output[outputWidth * outputHeight];
    // Load kernel source code (assuming it's in a file named "histogram.cl")
    FILE* file = fopen("convo_base.cl", "r");
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

    // Initialize OpenCL variables.
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Load the kernel source code into a program.
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, NULL);
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
    cl_kernel kernel = clCreateKernel(program, "convolution", NULL);

    // Create memory objects for image, filter, and output.
    cl_mem imageBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * imageWidth * imageHeight, NULL, NULL);
    cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * filterWidth * filterHeight, NULL, NULL);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * outputWidth * outputHeight, NULL, NULL);

    // Set kernel arguments.
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
    clSetKernelArg(kernel, 6, sizeof(int), &filterHeight);

    // Copy data to device buffers.
    clEnqueueWriteBuffer(queue, imageBuffer, CL_TRUE, 0, sizeof(int) * imageWidth * imageHeight, image, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, filterBuffer, CL_TRUE, 0, sizeof(int) * filterWidth * filterHeight, filter, 0, NULL, NULL);

    // Execute the kernel.
    size_t globalWorkSize[2] = {outputWidth, outputHeight};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(int) * outputWidth * outputHeight, output, 0, NULL, NULL);

    // Print the result.
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            printf("%d ", output[i*outputWidth  + j]);
        }
        printf("\n");
    }

    // Clean up OpenCL resources.
    clReleaseMemObject(imageBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
