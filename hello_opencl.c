#include <stdio.h>
#include <CL/cl.h>

int main() {
    // Get the number of OpenCL platforms available
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);

    if (numPlatforms == 0) {
        printf("No OpenCL platforms found.\n");
        return 1;
    }

    // Get platform information
    cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    printf("OpenCL Platforms:\n");
    for (cl_uint i = 0; i < numPlatforms; i++) {
        char platformName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
        printf("  Platform %d: %s\n", i + 1, platformName);

        // Get the number of devices for this platform
        cl_uint numDevices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

        if (numDevices == 0) {
            printf("    No OpenCL devices found for this platform.\n");
        } else {
            // Get device information
            cl_device_id* devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

            printf("    Devices:\n");
            for (cl_uint j = 0; j < numDevices; j++) {
                char deviceName[128];
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
                printf("      Device %d: %s\n", j + 1, deviceName);
            }

            free(devices);
        }
    }

    free(platforms);

    return 0;
}
