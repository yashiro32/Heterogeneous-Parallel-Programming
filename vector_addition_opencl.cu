#include <wb.h> //@@ wb include opencl.h for you

//@@ OpenCL Kernel


int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
	
  cl_int clerr = CL_SUCCESS;
	
  cl_platform_id cpPlatform;
  clerr = clGetPlatformIDs(1, &cpPlatform, NULL);
	
  cl_device_id cldevs;
  clerr = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cldevs, NULL);
	
  // cl_context clctx = clCreateContextFromType(0, CL_DEVICE_TYPE_ALL, NULL, NULL, &clerr);
  cl_context clctx = clCreateContext(0, 1, &cldevs, NULL, NULL, &clerr);
  
  size_t parmsz;
	
  clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);
	
  // cl_device_id *cldevs = (cl_device_id*)malloc(parmsz);
  clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);
  
  cl_command_queue clcmdq = clCreateCommandQueue(clctx, cldevs, 0, &clerr);
	
  cl_program clpgm;
	
  const char* kernelSource = "__kernel void vadd(__global const float* a, __global const float* b, __global float* c, int n) {\nint id = get_global_id(0);\nc[id] = a[id] + b[id];\n}";
  // size_t source_size = strlen(vaddsrc) * sizeof(char);
  
  clpgm = clCreateProgramWithSource(clctx, 1, (const char**)&kernelSource, NULL, &clerr);
  
  char clcompileflags[4096];
	
  sprintf(clcompileflags, "-cl-mad-enable");
	
  clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL);
  
  cl_kernel clkern = clCreateKernel(clpgm, "vadd", &clerr);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int size = inputLength * sizeof(float);
	
  cl_mem d_a, d_b, d_c;
  
  d_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY, size, NULL, NULL);
  d_b = clCreateBuffer(clctx, CL_MEM_READ_ONLY, size, NULL, NULL);
  d_c = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, size, NULL, NULL);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  clEnqueueWriteBuffer(clcmdq, d_a, CL_FALSE, 0, size, (const void*)hostInput1, 0, 0, NULL);
  clEnqueueWriteBuffer(clcmdq, d_b, CL_FALSE, 0, size, (const void*)hostInput2, 0, 0, NULL);
	
  // d_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hostInput1, NULL);
  // d_b = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hostInput2, NULL);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  size_t Bsz = 64;
  size_t Gsz = inputLength-1/(float)Bsz+1;
  
  clerr = clSetKernelArg(clkern, 0, sizeof(cl_mem), (void*)&d_a);
  clerr = clSetKernelArg(clkern, 1, sizeof(cl_mem), (void*)&d_b);
  clerr = clSetKernelArg(clkern, 2, sizeof(cl_mem), (void*)&d_c);
  clerr = clSetKernelArg(clkern, 3, sizeof(int), &inputLength);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  cl_event event = NULL;
  clerr = clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &Gsz, &Bsz, 0, NULL, &event);
  
  clerr = clWaitForEvents(1, &event);
  // cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  clEnqueueReadBuffer(clcmdq, d_c, CL_TRUE, 0, size, (void*)hostOutput, 0, NULL, NULL);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
