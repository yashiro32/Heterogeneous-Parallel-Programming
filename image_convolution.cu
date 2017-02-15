#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define BLOCK_WIDTH 32
#define EXTRA Mask_width-1
#define O_TILE_WIDTH 28

//@@ INSERT CODE HERE
__global__ void sharedConvolution_2D_kernel(float *in, float *out, const float* __restrict__ mask, int width, int height, int channels, int k){
	int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int row_o = blockIdx.y*O_TILE_WIDTH + ty;
    int col_o = blockIdx.x*O_TILE_WIDTH + tx;
    
    int row_i = row_o - Mask_radius;
    int col_i = col_o - Mask_radius;
    
    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
    
	// for (int k = 0; k < channels; k++) {
		
	
    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
        Ns[ty][tx] = in[(row_i*width + col_i)*channels+k];
    } else {
        Ns[ty][tx] = 0.0f;
    }
    
    __syncthreads();
     
    float output = 0.0f;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
        for (int i = 0; i < Mask_width; i++) {
			for (int j = 0; j < Mask_width; j++) {
				// float filter = mask[i*Mask_width+j];
			    // float pixel = Ns[i+ty][j+tx];
                // output += filter * pixel;
				output += mask[i*Mask_width+j] * Ns[i+ty][j+tx];
				
            }
        }
	  
        __syncthreads();
		
		
		if (row_o < height && col_o < width) {
            out[(row_o*width + col_o)*channels+k] = min(max(output, 0.0f), 1.0f);
		    // out[((row_o*width + col_o)*channels)+k] = output;
        }
		
    }
	
    
	// }
		
}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dimGrid((imageWidth-1)/O_TILE_WIDTH+1, (imageHeight-1)/O_TILE_WIDTH+1, 1);
	
	sharedConvolution_2D_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, imageWidth, imageHeight, imageChannels, 0);
	sharedConvolution_2D_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, imageWidth, imageHeight, imageChannels, 1);
	sharedConvolution_2D_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, imageWidth, imageHeight, imageChannels, 2);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
