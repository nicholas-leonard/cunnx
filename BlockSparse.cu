#define SOFTMAXTREE_THREADS 32

__global__ void cunnx_BlockSparse_updateOutput_kernel(
  float *output, float *input, float *inputIndices, float *outputIndices, 
  float *inputScales, float *outputScales, float *weight, float *bias,  
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize)
{
  __shared__ float buffer[SOFTMAXTREE_THREADS+1];
  __shared__ float output[SOFTMAXTREE_THREADS];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *input_k = input + k*inputWindowSize*inputSize;
  float *output_k = output + k*outputWindowSize*outputSize;
  float *inputIndices_k = inputIndices + k*inputWindowSize;
  float *outputIndices_k = outputIndices + k*outputWindowSize;
  float *inputScales_k = inputScales + k*inputWindowSize;
  float *outputScales_k = outputScales + k*outputWindowSize;
  
  float *blockOutput, *blockWeight, *blockBias;

  // loop through blocks
  for (int l=0; l<inputWindowSize; l++)
  {
    for (int m=0; m<outputWindowSize; m++)
    {
      /* Linear */
      nodeWeight = weight + parentIdx*nInput;
      nodeBias = bias + parentIdx;
      
      // addmv (dot products)
      for (int j=0; j<outputSize; j++)
      {
         // zero buffer
        buffer[tx] = 0;
        
        // multiply
        for (int i=tx; i<nInput; i+=i_step)
        {
          buffer[tx] += input_k[i]*nodeWeight[j*nInput + i];
          CudaAssert(isfinite(buffer[tx]))
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
          CudaAssert(isfinite(buffer[0]))
          linearOutput[j] = buffer[0] + nodeBias[j];
        }
      }
      
      __syncthreads();
      
      /* LogSoftMax */
      nodeOutput = logsoftOutput + maxFamilyPath*k + n;
      
      n += nChildren;
      CudaAssert((n <= maxFamilyPath))
      /* Break when root is reached */
      if (parentId == rootId) 
      {
        break;
      }
      childId = parentId;
    }
  }
  if (tx == 0) 
  {
    output[k] = narrowsum;
    CudaAssert(isfinite(narrowsum))
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
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndices = THCudaTensor_newContiguous(inputIndices);
  outputIndices = THCudaTensor_newContiguous(outputIndices); 
  inputScales = THCudaTensor_newContiguous(inputScales);
  outputScales = THCudaTensor_newContiguous(outputScales); 
  
  THCudaTensor_resize3d(output, input->[0], outputIndices->size[1], outputSize);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
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
  return 1;
}

static const struct luaL_Reg cunnx_SoftMaxTree__ [] = {
  {"BlockSparse_updateOutput", cunnx_SoftMaxTree_updateOutput},
  //{"BlockSparse_updateGradInput", cunnx_SoftMaxTree_updateGradInput},
  //{"BlockSparse_accGradParameters", cunnx_SoftMaxTree_accGradParameters},
  //{"BlockSparse_updateParameters", cunnx_SoftMaxTree_updateParameters},
  {NULL, NULL}
};

static void cunnx_BlockSparse_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_BlockSparse__, "nn");
  lua_pop(L,1);
}

