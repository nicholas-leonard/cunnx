#define BLOCKSPARSE_THREADS 32
#define BLOCKSPARSE_MAXOUTPUTSIZE 10000
  
__global__ void cunnx_BlockSparse_updateOutput_kernel(
  float *output, float *input, float *inputIndices, float *outputIndices, 
  float *inputScales, float *outputScales, float *weight, float *bias,  
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  __shared__ float outputBuffer[BLOCKSPARSE_MAXOUTPUTSIZE];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *input_k = input + k*inputWindowSize*inputSize;
  float *output_k = output + k*outputWindowSize*outputSize;
  float *inputIndices_k = inputIndices + k*inputWindowSize;
  float *outputIndices_k = outputIndices + k*outputWindowSize;
  float *inputScales_k = inputScales + k*inputWindowSize;
  float *outputScales_k = outputScales + k*outputWindowSize;
  
  // loop through blocks
  for (int m=0; m<outputWindowSize; m++)
  {
    int outputIdx = (int)outputIndices_k[m] - 1;
    float outputScale = outputScales_k[m];
    // break on non-positive scale. 
    if (outputScale <= 0) break;
      
    float *blockOutput = output_k + m*outputSize;
    float *blockBias = bias + outputIdx*outputSize;
    
    for (int j=tx; j<outputSize; j+=i_step)
    {
      outputBuffer[j] = blockBias[j];
    }
    
    for (int l=0; l<inputWindowSize; l++)
    {
      int inputIdx = (int)inputIndices_k[l] - 1;
      float inputScale = inputScales_k[l];
      // break on non-positive scale. 
      if (inputScale <= 0) 
        break;
      
      float *blockInput = input_k + l*inputSize;
      float *blockWeight = weight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;
      
      // addmv (dot products)
      for (int j=0; j<outputSize; j++)
      {
        // zero buffer
        buffer[tx] = 0;
        
        // multiply
        for (int i=tx; i<inputSize; i+=i_step)
        {
          buffer[tx] += inputScale*blockInput[i]*blockWeight[j*inputSize + i];
        }
        
        // add (reduce)
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
          __syncthreads();
          if (tx < stride)
            buffer[tx] += buffer[tx+stride];
        }
        
        if (tx == 0)
        {
          outputBuffer[j] += buffer[0];
        }
        
      }
      
    }
    
    __syncthreads();
      
    for (int j=tx; j<outputSize; j+=i_step)
    {
      blockOutput[j] = outputScale*outputBuffer[j];
    }
    
    __syncthreads();
    
  }
}


static int cunnx_BlockSparse_updateOutput(lua_State *L)
{ 
  /* input, inputIndices, outputIndices, inputScales, outputScales*/
  // batchSize x inputWindowSize x inputSize
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  // batchSize x inputWindowSize
  THCudaTensor *inputIndices = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *inputScales = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  // batchSize x outputWindowSize
  THCudaTensor *outputIndices = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *outputScales = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  
  // nOutputBlock x nInputBlock x outputSize x inputSize
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  // nOutputBlock x outputSize
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 3, 2, "3D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[2] == inputSize, 2, "invalid input size"); 
  luaL_argcheck(L, inputIndices->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndices->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScales->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScales->nDimension == 2, 6, "2D(batch mode) tensor expected");
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndices = THCudaTensor_newContiguous(inputIndices);
  outputIndices = THCudaTensor_newContiguous(outputIndices); 
  inputScales = THCudaTensor_newContiguous(inputScales);
  outputScales = THCudaTensor_newContiguous(outputScales); 
  
  THCudaTensor_resize3d(output, input->size[0], outputIndices->size[1], outputSize);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(input), 
    THCudaTensor_data(inputIndices), THCudaTensor_data(outputIndices),
    THCudaTensor_data(inputScales), THCudaTensor_data(outputScales),
    THCudaTensor_data(weight), THCudaTensor_data(bias),  
    inputSize, outputSize, nInputBlock, nOutputBlock,
    inputIndices->size[1], outputIndices->size[1]
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  THCudaTensor_free(input);
  THCudaTensor_free(inputIndices);
  THCudaTensor_free(outputIndices);
  THCudaTensor_free(inputScales);
  THCudaTensor_free(outputScales);
  return 1;
}

static const struct luaL_Reg cunnx_BlockSparse__ [] = {
  {"BlockSparse_updateOutput", cunnx_BlockSparse_updateOutput},
  //{"BlockSparse_updateGradInput", cunnx_BlockSparse_updateGradInput},
  //{"BlockSparse_accGradParameters", cunnx_BlockSparse_accGradParameters},
  //{"BlockSparse_updateParameters", cunnx_BlockSparse_updateParameters},
  {NULL, NULL}
};

static void cunnx_BlockSparse_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_BlockSparse__, "nn");
  lua_pop(L,1);
}
