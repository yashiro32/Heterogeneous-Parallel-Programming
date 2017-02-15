// Histogram Equalization
#include    <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
/**
converts image pixel from flaot to unsigned char.
each thread computes all three channels for a pixel.
**/
__global__ void convertToChar(float * in, unsigned char * out, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (index < size) {
		out[index] = (unsigned char) (in[index] * 255);
	}
}

/**
converts image to grayscale using a specific function
**/
__global__ void convertToGrayScale(unsigned char * ucharImg, unsigned char * grayImg, int width, int height, int channels)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
	int index = row*width + col;
	
	if (row < height && col < width) {
		unsigned char r = ucharImg[index*channels+0];
	    unsigned char g = ucharImg[index*channels+1];
	    unsigned char b = ucharImg[index*channels+2];
	
	    grayImg[index] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
	}
	
}


/**
computes the history equalization and converts image back to float
**/
__global__ void hist_eq(unsigned char * deviceCharImg, float * output, float* cdf, float cdfmin, int size)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
	
	int i = tx+blockDim.x*bx;
	
	if(i < size)
	{
		deviceCharImg[i] = min(max(255*(cdf[deviceCharImg[i]] - cdfmin)/(1 - cdfmin),0.0),255.0);
		
		output[i] = (float) (deviceCharImg[i]/255.0);
		
	}
}


/**
*computes the histogram of an image 
@param unsigned char * buffer: takes image converted to unsigned char
@param unsigned int * histo: takes histogram of an image
@param long size: takes total pixels of image
**/
__global__ void histo_kernel(unsigned char * buffer, unsigned int * histo, long size)
{
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

/*
probability function of a pixel that returns a float.
@param int x: takes pixel value;
@param int width: width of image
@param int height: height of image
*/
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
	//  device variables
	float * deviceInputImageData;
	float * deviceOutputImageData;
	unsigned char * deviceUCharImage;
	unsigned char * deviceGrayImg;
	
	float *deviceHistoScan;
	float *hostHistoScan;
	
	
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
	int imageDataSize = imageWidth * imageHeight * imageChannels;
	int imageSize = imageWidth * imageHeight;
	
	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
	
	//allocate memory for device variables
	cudaMalloc((void **) &deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
	cudaMalloc((void **) &deviceUCharImage, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
	cudaMalloc((void **) &deviceGrayImg, imageWidth*imageHeight*sizeof(unsigned char));
	
	cudaMemcpy(deviceInputImageData, 
			   hostInputImageData, 
			   imageWidth*imageHeight*imageChannels*sizeof(float), 
			   cudaMemcpyHostToDevice);
	
    wbLog(TRACE, "image width: ",imageWidth,", image height: ",imageHeight);
	
    //@@ insert code here
	dim3 dimBlock(12, 12, 1);
	dim3 dimGrid((imageWidth - 1)/12 + 1, (imageHeight - 1)/12 + 1, 1);
	
	//convert the image to unsigned char
	convertToChar<<<(imageDataSize-1)/1024+1, 1024>>>(deviceInputImageData, deviceUCharImage, imageDataSize);
	
		
	//  need to convert image to grayscale
	convertToGrayScale<<<dimGrid, dimBlock>>>(deviceUCharImage, deviceGrayImg, imageWidth, imageHeight, imageChannels);
	
	//cudaMemcpy(hostGrayImg, deviceGrayImg, imageWidth*imageHeight*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	//  allocate histogram host and set initial values of array to zero.
	unsigned int * hostHistogram;
	hostHistogram = (unsigned int *) malloc(HISTOGRAM_LENGTH*sizeof(unsigned int));
	
	memset(hostHistogram, 0, sizeof(unsigned int) * HISTOGRAM_LENGTH);
	
	//  allocation for histogram from host to device
	unsigned int * deviceHistogram;
	cudaMalloc((void **) &deviceHistogram,HISTOGRAM_LENGTH*sizeof(unsigned int));
	cudaMemset(deviceHistogram, 0, sizeof(unsigned int) * HISTOGRAM_LENGTH);
	
	//compute the histogram
	histo_kernel<<<(imageSize-1)/1024+1, 1024>>>(deviceGrayImg, deviceHistogram, imageSize);	
	
	//copy result back to host histogram
	cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	//  compute scan operation for histogram
	// float * hostCDF;
	// hostCDF = (float *)malloc(HISTOGRAM_LENGTH*sizeof(float));
	
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
	
	//  copy host cdf to device
	cudaMalloc((void **) &deviceHistoScan, HISTOGRAM_LENGTH*sizeof(float));
	cudaMemcpy(deviceHistoScan, hostHistoScan, sizeof(float) * HISTOGRAM_LENGTH, cudaMemcpyHostToDevice);
	
	//  histogram equalization function
	hist_eq<<<(imageDataSize-1)/1024+1, 1024>>>(deviceUCharImage, deviceOutputImageData, deviceHistoScan, cdfmin, imageDataSize);
	
	//  copy results back to host
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, sizeof(float) * imageDataSize, cudaMemcpyDeviceToHost);
	wbSolution(args, outputImage);
	
	cudaFree(deviceUCharImage);
	cudaFree(deviceGrayImg);
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	
	free(hostInputImageData);
	free(hostOutputImageData);
	
	wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    
	return 0;
}