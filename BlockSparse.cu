#define BLOCKSPARSE_THREADS 32
#define BLOCKSPARSE_MAXBLOCKSIZE 10000
  
__global__ void cunnx_BlockSparse_updateOutput_kernel(
  float *output, float *input, float *inputIndice, float *outputIndice, 
  float *inputScale, float *outputScale, float *weight, float *bias,  
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  __shared__ float outputBuffer[BLOCKSPARSE_MAXBLOCKSIZE];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *input_k = input + k*inputWindowSize*inputSize;
  float *output_k = output + k*outputWindowSize*outputSize;
  float *inputIndice_k = inputIndice + k*inputWindowSize;
  float *outputIndice_k = outputIndice + k*outputWindowSize;
  float *inputScale_k = inputScale + k*inputWindowSize;
  float *outputScale_k = outputScale + k*outputWindowSize;
  
  // loop through blocks
  for (int m=0; m<outputWindowSize; m++)
  {
    int outputIdx = (int)outputIndice_k[m] - 1;
    float outputScale = outputScale_k[m];
    
    if (outputScale <= 0) // break on non-positive scale. 
      break;
      
    float *blockOutput = output_k + m*outputSize;
    float *blockBias = bias + outputIdx*outputSize;
    
    for (int j=tx; j<outputSize; j+=i_step)
    {
      outputBuffer[j] = blockBias[j];
    }
    
    for (int l=0; l<inputWindowSize; l++)
    {
      int inputIdx = (int)inputIndice_k[l] - 1;
      float inputScale = inputScale_k[l];
      
      if (inputScale <= 0) // break on non-positive scale. 
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
  /* input, inputIndice, outputIndice, inputScale, outputScale*/
  // batchSize x inputWindowSize x inputSize
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  // batchSize x inputWindowSize
  THCudaTensor *inputIndice = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *inputScale = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  // batchSize x outputWindowSize
  THCudaTensor *outputIndice = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *outputScale = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  
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
  luaL_argcheck(L, inputIndice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndice = THCudaTensor_newContiguous(inputIndice);
  outputIndice = THCudaTensor_newContiguous(outputIndice); 
  inputScale = THCudaTensor_newContiguous(inputScale);
  outputScale = THCudaTensor_newContiguous(outputScale); 
  
  THCudaTensor_resize3d(output, input->size[0], outputIndice->size[1], outputSize);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(input), 
    THCudaTensor_data(inputIndice), THCudaTensor_data(outputIndice),
    THCudaTensor_data(inputScale), THCudaTensor_data(outputScale),
    THCudaTensor_data(weight), THCudaTensor_data(bias),  
    inputSize, outputSize, nInputBlock, nOutputBlock,
    inputIndice->size[1], outputIndice->size[1]
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  THCudaTensor_free(input);
  THCudaTensor_free(inputIndice);
  THCudaTensor_free(outputIndice);
  THCudaTensor_free(inputScale);
  THCudaTensor_free(outputScale);
  return 1;
}


__global__ void cunnx_BlockSparse_updateGradInput_kernel(
  float *gradInput, float* gradOutputScale, float *gradOutput, 
  float *output, float *inputIndice, float *outputIndice, 
  float *inputScale, float *outputScale, float *weight,  
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  __shared__ float gradInputBuffer[BLOCKSPARSE_MAXBLOCKSIZE];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *gradInput_k = gradInput + k*inputWindowSize*inputSize;
  float *gradOutput_k = gradOutput + k*outputWindowSize*outputSize;
  float *output_k = output + k*outputWindowSize*outputSize;
  float *inputIndice_k = inputIndice + k*inputWindowSize;
  float *outputIndice_k = outputIndice + k*outputWindowSize;
  float *inputScale_k = inputScale + k*inputWindowSize;
  float *outputScale_k = outputScale + k*outputWindowSize;
  
  // 1. loop through blocks to get gradInput
  for (int l=0; l<inputWindowSize; l++)
  {
    int inputIdx = (int)inputIndice_k[l] - 1;
    float inputScale = inputScale_k[l];
    
    if (inputScale <= 0) // break on non-positive scale. 
      break;
    
    float *blockGradInput = gradInput_k + l*inputSize;
      
    for (int j=tx; j<inputSize; j+=i_step)
    {
      gradInputBuffer[j] = 0;
    }
    
    for (int m=0; m<outputWindowSize; m++)
    {
      int outputIdx = (int)outputIndice_k[m] - 1;
      float outputScale = outputScale_k[m];
      
      if (outputScale <= 0) // break on non-positive scale. 
        break;
        
      float *blockGradOutput = gradOutput_k + m*outputSize;     
      float *blockWeight = weight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;
      
      // addmv (dot products)
      for (int i=tx; i<inputSize; i+=i_step)
      {
       // zero buffer
        buffer[tx] = 0;
        
        for (int j=0; j<outputSize; j++)
        {
          // multiply
          buffer[tx] += blockGradOutput[j]*blockWeight[j*inputSize + i];
        }
        
        // accumulate 
        gradInputBuffer[i] += buffer[tx];
      }
      
    }
    
    __syncthreads();
      
    for (int i=tx; i<inputSize; i+=i_step)
    {
      blockGradInput[i] = gradInputBuffer[i];
    }
    
    __syncthreads();
    
  }
  
  // 2. get gradients for outputScale (to be backwarded to a Gater)
  for (int m=0; m<outputWindowSize; m++)
  {
    float outputScale = outputScale_k[m];
    // break on non-positive scale. 
    if (outputScale <= 0) break;
      
    float *blockGradOutput = gradOutput_k + m*outputSize;
    float *blockOutput = output_k + m*outputSize;
    
    buffer[tx] = 0;
    
    for (int j=tx; j<outputSize; j+=i_step)
    {
      buffer[tx] += blockOutput[j]*blockGradOutput[j];
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
      gradOutputScale_k[m] += buffer[0];
    }
  }
}


static int cunnx_BlockSparse_updateGradInput(lua_State *L)
{ 
  /* input, inputIndice, outputIndice, inputScale, outputScale*/
  // batchSize x inputWindowSize x inputSize
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  // batchSize x inputWindowSize
  THCudaTensor *inputIndice = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *inputScale = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  // batchSize x outputWindowSize
  THCudaTensor *outputIndice = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *outputScale = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 7, "torch.CudaTensor");
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  
  // nOutputBlock x nInputBlock x outputSize x inputSize
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *gradOutputScale = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradOutputScale", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 3, 2, "3D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[2] == inputSize, 2, "invalid input size"); 
  luaL_argcheck(L, inputIndice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndice = THCudaTensor_newContiguous(inputIndice);
  outputIndice = THCudaTensor_newContiguous(outputIndice); 
  inputScale = THCudaTensor_newContiguous(inputScale);
  outputScale = THCudaTensor_newContiguous(outputScale); 
  gradOutput = THCudaTensor_newContiguous(gradOutput);
  
  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_resizeAs(gradOutputScale, outputScale);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(gradInput), THCudaTensor_data(gradOutputScale), 
    THCudaTensor_data(gradOutput), THCudaTensor_data(inputIndice), 
    THCudaTensor_data(outputIndice), THCudaTensor_data(inputScale), 
    THCudaTensor_data(outputScale), THCudaTensor_data(weight), 
    inputSize, outputSize, nInputBlock, nOutputBlock,
    inputIndice->size[1], outputIndice->size[1]
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  THCudaTensor_free(input);
  THCudaTensor_free(inputIndice);
  THCudaTensor_free(outputIndice);
  THCudaTensor_free(inputScale);
  THCudaTensor_free(outputScale);
  THCudaTensor_free(gradOutput);
  return 2;
}

static const struct luaL_Reg cunnx_BlockSparse__ [] = {
  {"BlockSparse_updateOutput", cunnx_BlockSparse_updateOutput},
  {"BlockSparse_updateGradInput", cunnx_BlockSparse_updateGradInput},
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
