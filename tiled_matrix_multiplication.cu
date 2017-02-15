#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

int const TILE_WIDTH = 16;

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH]; 
	
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
	
  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
	
  float Cvalue = 0;
	
  int m = numCRows; // Rows of Matrix C which is equal to the row of Matrix A
  int k = numCColumns; // Columns of Matrix C which is equal to the columns of Matrix B
  int n = numAColumns; // Columns of Matrix A which is equal to rows of Matrix B
	
  // Loop over the A and B tiles required to compute the C element
	for (int t = 0; t < (n-1)/TILE_WIDTH + 1; ++t) {
		// Collaborative loading of A and B tiles into shared memory
		if ((Row < m) && (t*TILE_WIDTH+tx < n)) {
			ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH+tx];
		} else {
			ds_A[ty][tx] = 0.0;
		}
		
		if ((t*TILE_WIDTH+ty < n) && (Col < k)) {
			ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*k + Col];
		} else {
			ds_B[ty][tx] = 0.0;
		}
		
		__syncthreads();
		
		for (int i = 0; i < TILE_WIDTH; ++i) {
			Cvalue += ds_A[ty][i] * ds_B[i][tx];
		}
		
		__syncthreads();
	}
	
	if (Row < m && Col < k) {
		C[Row*k + Col] = Cvalue;
	}
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float*)malloc(sizeof(float) * (numCRows * numCColumns));

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, sizeof(float) * (numARows * numAColumns));
  cudaMalloc((void**)&deviceB, sizeof(float) * (numBRows * numBColumns));
  cudaMalloc((void**)&deviceC, sizeof(float) * (numCRows * numCColumns));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float) * (numARows * numAColumns), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * (numBRows * numBColumns), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numCColumns-1)/TILE_WIDTH+1, (numCRows-1)/TILE_WIDTH+1, 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeof(float) * (numCRows * numCColumns), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
