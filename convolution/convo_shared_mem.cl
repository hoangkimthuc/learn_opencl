__kernel void convolution(
    __global int* image, 
    __global int* filter,
    __global int* output,
    int imageWidth,
    int imageHeight,
    int filterWidth,
    int filterHeight) {

    int x = get_global_id(0);
    int y = get_global_id(1);
    int outputWidth = imageWidth - filterWidth + 1;    

    for (int i = 0; i < filterHeight; i++) {
        for (int j = 0; j < filterWidth; j++) {
            output[x*outputWidth + y] += image[(x+i)*imageWidth +y+j] * filter[i*filterWidth + j];
        }
    }
}
        