#define WINDOWGATE_THREADS 128

__global__ void cunnx_WindowGate_updateOutput_kernel(
  float *output, float *inputIndiceCuda, float *outputIndiceCuda,
  const float* input, int inputSize, int outputSize, int inputWindowSize, 
  int outputWindowSize, int windowStride, int softmax)
{
  __shared__ float buffer[WINDOWGATE_THREADS+1];
  unsigned int tx = threadIdx.x;
  unsigned int k = blockIdx.x;
  const float *input_k = input + inputSize*k;
  float *output_k = output + outputWindowSize*k;
  int inputIdx;
  int outputIdx;
  
  // get coordinate of centoid
  buffer[tx] = 0;
  for (unsigned int i=tx; i<inputSize; i+=blockDim.x)
    buffer[tx] += i*input_k[i];
  
  // add (reduce)
  
  for (unsigned int stride = WINDOWGATE_THREADS >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  
  if (tx == 0)
  {
    float centroid = buffer[0];
    
    if (!softmax) 
      centroid /=inputSize;
    
    // inputIdx is left corner of window
    int inputIdx = centroid - (float)inputWindowSize*0.5;
    
    // make centroid a number between 0 and 1
    centroid /=inputSize;
    
    int outputIdx = centroid/(float)(inputSize*outputSize) - (float)outputWindowSize*0.5;

    // clip indices
    inputIdx = fminf(inputIdx, inputSize-inputWindowSize);
    inputIdx = fmaxf(inputIdx, 1);
    outputIdx = fminf(outputIdx, outputSize-outputWindowSize);
    outputIdx = fmaxf(outputidx, 1);
    
    inputIndiceCuda[k] = inputIdx;
    outputIndiceCuda[k] = outputIdx;
    buffer[0] = inputIdx;
  }
  
  __syncthreads();
  
  inputIdx = buffer[0] - 1;
  outputIdx = buffer[WINDOWGATE_THREADS] - 1;
  
  float *inputWindow = input_k + inputIdx;
  for (int i=tx; i<outputWindowSize; i+=blockDim.x)
    output_k[i] = inputWindow[(int)floorf(((float)i)/stride)];

}

  
static int cunnx_WindowGate_updateOutput(lua_State *L)
{ 
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int inputWindowSize = luaT_getfieldcheckint(L, 1, "inputWindowSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
  int windowStride = luaT_getfieldcheckint(L, 1, "windowStride");
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int softmax = luaT_getfieldcheckboolean(L, 1, "softmax");
  
  THCudaTensor *inputIndiceCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "inputIndiceCuda", "torch.CudaTensor");
  THLongTensor *inputIndice = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "inputIndice", "torch.LongTensor");
  THCudaTensor *outputIndiceCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceCuda", "torch.CudaTensor");
  THLongTensor *outputIndice = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndice", "torch.LongTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  
  THCudaTensor_resize1d(output, batchSize, outputWindowSize);
  THCudaTensor_resize1d(inputIndiceCuda, batchSize);
  THCudaTensor_resize1d(outputIndiceCuda, batchSize);
    
  /* call cudakernel */
  dim3 blocks(batchSize); // each cuda-block is an example
  dim3 threads(WINDOWGATE_THREADS);
  cunnx_WindowGate_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(inputIndiceCuda),
    THCudaTensor_data(outputIndiceCuda),
    (const float*)THCudaTensor_data(input), inputSize, outputSize,
    inputWindowSize, outputWindowSize, windowStride, softmax
  );
  
  THLongTensor_copyCuda(inputIndice, inputIndiceCuda);
  THLongTensor_copyCuda(outputIndice, outputIndiceCuda);

  return 0;
}


static const struct luaL_Reg cunnx_WindowGate__ [] = {
  {"WindowGate_updateOutput", cunnx_WindowGate_updateOutput},
  {"WindowGate_updateGradInput", cunnx_WindowGate_updateGradInput},
  {NULL, NULL}
};

static void cunnx_WindowGate_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_WindowGate__, "nn");
  lua_pop(L,1);
}
