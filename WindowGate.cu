#define WINDOWGATE_THREADS 128

__global__ void cunnx_WindowGate_updateOutput_kernel(
  float *output, float *outputIndice, const float* input, 
  int inputWindowSize, int outputWindowSize, int inputSize, int outputSize,
  int softmax)
{
  __shared__ float buffer[WINDOWGATE_THREADS];
  unsigned int tx = threadIdx.x;
  unsigned int k = blockIdx.x;
  const float *input_k = input + inputSize*k;
  float *output_k = output + outputWindowSize*k;
  
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
  
  __syncthreads();
  float centroid = buffer[0];
  
  if (!softmax) 
    centroid /=inputSize;
  
  // make centroids a number between 0 and 1
  centroid /=inputSize;
  
  self._inputIndice:mul(self.centroid, input:size(2)):add(-self.inputWindowSize*0.5)
  self.inputIndice:copy(self._inputIndice)
  self._outputIndice:mul(self.centroid, self.outputSize):add(-self.outputWindowSize*0.5)
  self.outputIndice:copy(self._outputIndice)
}

  
static int cunnx_WindowGate_updateOutput(lua_State *L)
{ 
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int inputWindowSize = luaT_getfieldcheckint(L, 1, "inputWindowSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int softmax = luaT_getfieldcheckboolean(L, 2, "softmax");
  
  THCudaTensor *outputIndiceCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceCuda", "torch.CudaTensor");
  THLongTensor *outputIndice = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndice", "torch.LongTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
    
  /* call cudakernel */
  dim3 blocks(batchSize); // each cuda-block is an example
  dim3 threads(WINDOWGATE_THREADS);
  cunnx_WindowGate_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(outputIndiceCuda),
    (const float*)THCudaTensor_data(input), 
    inputWindowSize, outputWindowSize, inputSize, outputSize,
    softmax
  );
  

  return 1;
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
