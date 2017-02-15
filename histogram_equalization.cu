// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

#define BLOCK_SIZE 128

//@@ insert code here
	
__device__ unsigned char clamp(unsigned char x, unsigned char start, unsigned char end) {
    return min(max(x, start), end);
}

__global__ void convertFloatToChar(float *in, unsigned char *out, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (index < size) {
		out[index] = (unsigned char) (in[index] * 255);
	}
}

__global__ void RGBToGrayScale(unsigned char *rgb, unsigned char *gray, int size, int width, int height, int channels) {
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
	int index = row*width + col;
	
	if (row < height && col < width) {
		unsigned char r = rgb[index*channels+0];
	    unsigned char g = rgb[index*channels+1];
	    unsigned char b = rgb[index*channels+2];
	
	    gray[index] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
	}
	
}

__global__ void histogram(unsigned char *buffer, unsigned int *histo, int size) {
	/* int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ unsigned int histo_private[256];
	
	if (threadIdx.x < 256) {
		histo_private[threadIdx.x] = 0;
	}
	
	__syncthreads();
	
	// Stride is total number of threads
	int stride = blockDim.x * gridDim.x;
	
	while (i < size) {
		atomicAdd(&(histo_private[buffer[i]]), 1);
		i += stride;
	}
	
	__syncthreads();
	
	if (threadIdx.x < 256) {
		atomicAdd(&histo[threadIdx.x], histo_private[threadIdx.x]);
	} */
	
	
	//  compute histogram with a private version in each block
	__shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
	
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
	//  index of current pixel
	int index = tx+bx*blockDim.x;
	
	//  set initial values of histogram to zero 
	if (tx < HISTOGRAM_LENGTH) histo_private[tx] = 0;
	
	__syncthreads();
	
	
	int stride = blockDim.x*gridDim.x;
	
	//iterate to add values
	while (index < stride)
	{
		atomicAdd(&(histo_private[buffer[index]]), 1);
		index += stride;
	}
	
	__syncthreads();
	
	//copy private histogram to device histogram
	if(tx<256)
	{
		atomicAdd(&(histo[tx]), histo_private[tx]);
	}
	
}

__global__ void scan(unsigned int * input, unsigned int * output, int len) {
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	unsigned int t = threadIdx.x;
	int i = start + t;
	
	__shared__ unsigned int XY[2*BLOCK_SIZE];
	
	if (start + t < len) {
		XY[t] = input[start + t];
	} else {
		XY[t] = 0.0f;
	}
	
	if (start + blockDim.x + t < len) {
		XY[t + blockDim.x] = input[start + blockDim.x + t];
	} else {
		XY[t + blockDim.x] = 0.0f;
	}
	
	__syncthreads();
	
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		int index = (threadIdx.x+1)*stride*2 - 1;
		if (index < 2*BLOCK_SIZE) {
			XY[index] += XY[index-stride];
		}
		
		__syncthreads();
	}
	
	for (int stride = BLOCK_SIZE/2; stride < 0; stride /= 2) {
		__syncthreads();
		
		int index = (threadIdx.x+1)*stride*2 - 1;
		
		if (index + stride < 2*BLOCK_SIZE) {
			XY[index+stride] += XY[index];
		}
	}
	
	__syncthreads();
	
	if (i < len) {
		output[i] = XY[t];
		output[start+blockDim.x+t] = XY[t+blockDim.x];
	}
	
}

__global__ void correct_color(unsigned char *uCharImage, float *cdf, float cdfmin, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (index < size) {
		// unsigned char val = uCharImage[index];
	    // uCharImage[index] = clamp(255*(cdf[val] - cdfmin)/(1 - cdfmin), 0, 255);
		uCharImage[index] = min(max(255*(cdf[uCharImage[index]] - cdfmin)/(1 - cdfmin),0.0),255.0);
	}
}

__global__ void convertCharToFloat(unsigned char* input, float *output, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (index < size) {
		output[index] = (float) (input[index]/255.0);
	}
}


float prob(int x, int width, int height)
{
	return 1.0*x/(width*height);
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
	float *deviceInputImageData;
	float *deviceOutputImageData;
	
	unsigned char *deviceCastImageData;
	unsigned char *deviceGrayScaleData;
	
	unsigned int *deviceHistogram;
	float *deviceHistoScan;
	
	unsigned int *hostHistogram;
	float *hostHistoScan;
	unsigned char *hostCastImageData;
	

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
	
	cudaMalloc((void**)&deviceInputImageData, sizeof(float) * imageWidth * imageHeight * imageChannels);
	cudaMalloc((void**)&deviceOutputImageData, sizeof(float) * imageWidth * imageHeight * imageChannels);
	
	cudaMalloc((void**)&deviceCastImageData, sizeof(unsigned char) * imageWidth * imageHeight * imageChannels);
	cudaMalloc((void**)&deviceGrayScaleData, sizeof(unsigned char) * imageWidth * imageHeight);
	cudaMalloc((void**)&deviceHistogram, sizeof(unsigned int) * HISTOGRAM_LENGTH);
	cudaMalloc((void**)&deviceHistoScan, sizeof(float) * HISTOGRAM_LENGTH);
	
	cudaMemcpy(deviceInputImageData, hostInputImageData, sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyHostToDevice);
	
	cudaMemset(deviceHistogram, 0, sizeof(unsigned int) * HISTOGRAM_LENGTH);
	cudaMemset(deviceHistoScan, 0.0f, sizeof(float) * HISTOGRAM_LENGTH);
	
	int imageDataSize = imageWidth * imageHeight * imageChannels;
	int imageSize = imageWidth * imageHeight;
	
	convertFloatToChar<<<(imageDataSize-1)/1024+1, 1024>>>(deviceInputImageData, deviceCastImageData, imageDataSize);

	dim3 dimBlock(1024, 1024);
	dim3 dimGrid((imageWidth-1)/1024+1, (imageHeight-1)/1024+1);
	
	RGBToGrayScale<<<dimGrid, dimBlock>>>(deviceCastImageData, deviceGrayScaleData, imageSize, imageWidth, imageHeight, imageChannels);
    	
	histogram<<<(imageSize-1)/256+1, 256>>>(deviceGrayScaleData, deviceHistogram, imageSize);
	
	hostHistogram = (unsigned int*) malloc(sizeof(unsigned int) * HISTOGRAM_LENGTH);
	memset(hostHistogram, 0, sizeof(unsigned int) * HISTOGRAM_LENGTH);
	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost);
	
	hostHistoScan = (float*) malloc(sizeof(float) * HISTOGRAM_LENGTH);
    memset(hostHistoScan, 0.0f, sizeof(float) * HISTOGRAM_LENGTH);
	
	hostHistoScan[0] = prob(hostHistogram[0], imageWidth, imageHeight);
	for (int i = 1; i < 256; i++) {
		hostHistoScan[i] = hostHistoScan[i-1] + prob(hostHistogram[i], imageWidth, imageHeight);
	}
	
	float cdfmin = hostHistoScan[0];
	for (int i = 1; i < 256; i++) {
		cdfmin = min(cdfmin, hostHistoScan[i]);
	}
	
	
	/* hostCastImageData = (unsigned char*) malloc(sizeof(unsigned char) * imageDataSize);
	cudaMemcpy(hostCastImageData, deviceCastImageData, sizeof(unsigned char) * imageDataSize, cudaMemcpyDeviceToHost);
	for (int i = 0; i < imageDataSize; i++) {
		unsigned char val = hostCastImageData[i];
		hostCastImageData[i] = clamp(255*(hostHistoScan[val] - cdfmin)/(1 - cdfmin), 0, 255);
	}
	
	for (int i = 0; i < imageDataSize; i++) {
		hostOutputImageData[i] = (float) (hostCastImageData[i]/255.0);
	} */
	
	
	
	cudaMemcpy(deviceHistoScan, hostHistoScan, sizeof(float) * HISTOGRAM_LENGTH, cudaMemcpyHostToDevice);
	correct_color<<<(imageDataSize-1)/256+1, 256>>>(deviceCastImageData, deviceHistoScan, cdfmin, imageDataSize);
	
	convertCharToFloat<<<(imageDataSize-1)/256+1, 256>>>(deviceCastImageData, deviceOutputImageData, imageDataSize);
	// cudaMemcpy(hostOutputImageData, deviceOutputImageData, sizeof(float) * imageDataSize, cudaMemcpyDeviceToHost);
	
    wbSolution(args, outputImage);

    //@@ insert code here
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceCastImageData);
	cudaFree(deviceGrayScaleData);
	cudaFree(deviceHistogram);
	cudaFree(deviceHistoScan);
	
	free(hostHistogram);
	free(hostHistoScan);
	// free(hostCastImageData);
	
	wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

