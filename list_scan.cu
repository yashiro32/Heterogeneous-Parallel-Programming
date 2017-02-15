// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void scan(float * input, float * output, float *block_total, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	unsigned int t = threadIdx.x;
	int i = start + t;
	
	__shared__ float XY[2*BLOCK_SIZE];
	
	if (start+t < len) {
		XY[t] = input[start+t];
	} else {
		XY[t] = 0.0f;
	}
	
	if (start + blockDim.x + t < len) {
		XY[t + blockDim.x] = input[start + blockDim.x + t];
	} else {
		XY[t + blockDim.x] = 0.0f;
	}
	
	__syncthreads();
	
	for (int stride = 1; stride <= BLOCK_SIZE; stride*=2) {
		int index = (threadIdx.x+1)*stride*2-1;
		if (index < 2*BLOCK_SIZE) {
			XY[index] += XY[index-stride];
		}
		__syncthreads();
    }
	
	for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
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
	
	if (t == blockDim.x-1) {
		block_total[blockIdx.x] = XY[2*blockDim.x-1];
	}
}

__global__ void addition(float *output, float *block_total) {
	if (blockIdx.x > 0) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	    int block_offset = blockIdx.x-1;
	
	    float extra = block_total[block_offset];
	
	    output[index] += extra;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
	
    float * deviceInput;
    float * deviceOutput;
	
	float *deviceInputBlockTotal;
	float *deviceOutputBlockTotal;
	float *deviceInputBlockTotal2;
	
    int numElements; // number of elements in the list
	
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
	
	int firstGridSize = ((numElements/2)-1)/BLOCK_SIZE+1;
	int secondGridSize = ((firstGridSize/2)-1)/BLOCK_SIZE+1;
	
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceInputBlockTotal, firstGridSize * sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceOutputBlockTotal, firstGridSize * sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceInputBlockTotal2, secondGridSize * sizeof(float)));
	
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
	wbCheck(cudaMemset(deviceInputBlockTotal, 0, firstGridSize * sizeof(float)));
	wbCheck(cudaMemset(deviceOutputBlockTotal, 0, firstGridSize * sizeof(float)));
	wbCheck(cudaMemset(deviceInputBlockTotal2, 0, secondGridSize * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimBlock3(1024, 1, 1);
		
	dim3 dimGrid(firstGridSize, 1, 1);
	dim3 dimGrid2(secondGridSize, 1, 1);
	dim3 dimGrid3((numElements-1)/1024+1, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceInputBlockTotal, numElements);
	cudaDeviceSynchronize();
	
	scan<<<dimGrid2, dimBlock>>>(deviceInputBlockTotal, deviceOutputBlockTotal, deviceInputBlockTotal2, firstGridSize);
	cudaDeviceSynchronize();
	
	addition<<<dimGrid3, dimBlock3>>>(deviceOutput, deviceOutputBlockTotal);
    cudaDeviceSynchronize();
	
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
	cudaFree(deviceInputBlockTotal);
	cudaFree(deviceOutputBlockTotal);
	cudaFree(deviceInputBlockTotal2);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
