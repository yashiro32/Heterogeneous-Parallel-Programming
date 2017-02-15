#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < len) {
		out[index] = in1[index] + in2[index];
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;
	
	/* float *h_A;
	float *h_B;
	float *h_C; */
	
	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	
	float *d_A0, *d_B0, *d_C0; // Device memory for stream 0
	float *d_A1, *d_B1, *d_C1; // Device memory for stream 1
	float *d_A2, *d_B2, *d_C2; // Device memory for stream 2
	float *d_A3, *d_B3, *d_C3; // Device memory for stream 3
	
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
	
	int SegSize = inputLength / 4;
	
	/* cudaHostAlloc((void**)&h_A, sizeof(float) * inputLength, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_B, sizeof(float) * inputLength, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_C, sizeof(float) * inputLength, cudaHostAllocDefault);
	
	cudaMemcpy(h_A, hostInput1, sizeof(float) * inputLength, cudaMemcpyHostToHost);
	cudaMemcpy(h_B, hostInput2, sizeof(float) * inputLength, cudaMemcpyHostToHost);
	cudaMemcpy(h_C, hostOutput, sizeof(float) * inputLength, cudaMemcpyHostToHost); */
	
	cudaMalloc((void**)&d_A0, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_B0, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_C0, sizeof(float) * SegSize);
	
	cudaMalloc((void**)&d_A1, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_B1, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_C1, sizeof(float) * SegSize);
	
	cudaMalloc((void**)&d_A2, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_B2, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_C2, sizeof(float) * SegSize);
	
	cudaMalloc((void**)&d_A3, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_B3, sizeof(float) * SegSize);
	cudaMalloc((void**)&d_C3, sizeof(float) * SegSize);
	
	dim3 dimBlock(256, 1, 1);
	dim3 dimGrid((SegSize-1)/256+1, 1, 1);
	
	for (int i = 0; i < inputLength; i += SegSize*4) {
		cudaMemcpyAsync(d_A0, hostInput1+i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, hostInput2+i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
		
		
		
		cudaMemcpyAsync(d_A1, hostInput1+i+SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_B1, hostInput2+i+SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
		
		
		
		cudaMemcpyAsync(d_A2, hostInput1+i+SegSize*2, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_B2, hostInput2+i+SegSize*2, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream2);
		
		
		
		cudaMemcpyAsync(d_A3, hostInput1+i+SegSize*3, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(d_B3, hostInput2+i+SegSize*3, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream3);
		
		
		
		vecAdd<<<dimGrid, dimBlock, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
		vecAdd<<<dimGrid, dimBlock, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);
		vecAdd<<<dimGrid, dimBlock, 0, stream2>>>(d_A2, d_B2, d_C2, SegSize);
		vecAdd<<<dimGrid, dimBlock, 0, stream3>>>(d_A3, d_B3, d_C3, SegSize);
		
		
		cudaMemcpyAsync(hostOutput+i, d_C0, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(hostOutput+i+SegSize, d_C1, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
		cudaMemcpyAsync(hostOutput+i+SegSize*2, d_C2, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream2);
		cudaMemcpyAsync(hostOutput+i+SegSize*3, d_C3, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream3);
		
	}

	// cudaMemcpy(hostOutput, h_C, sizeof(float) * inputLength, cudaMemcpyHostToHost);

    wbSolution(args, hostOutput, inputLength);
	
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);
	
	cudaFree(d_A0);
	cudaFree(d_B0);
	cudaFree(d_C0);
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);
	cudaFree(d_A2);
	cudaFree(d_B2);
	cudaFree(d_C2);
	cudaFree(d_A3);
	cudaFree(d_B3);
	cudaFree(d_C3);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
	
	/* free(h_A);
	free(h_B);
	free(h_C); */

    return 0;
}
