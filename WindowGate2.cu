#include "utils.h"
#define WINDOWGATE2_THREADS 128

__global__ void cunnx_WindowGate2_updateOutput_kernel(
  float *output, float *centroids, float *normalizedCentroids, 
  float *inputIndice, float *outputIndice,
  const float *input, const float *noise, int inputSize, int outputSize, 
  int inputWindowSize, int outputWindowSize, int windowStride, int train)
{
  __shared__ float buffer[WINDOWGATE2_THREADS+1];
  unsigned int tx = threadIdx.x;
  unsigned int k = blockIdx.x;
  const float *input_k = input + inputSize*k;
  float *output_k = output + outputWindowSize*k;
  
  // get coordinate of centoid
  buffer[tx] = 0;
  for (unsigned int i=tx; i<inputSize; i+=blockDim.x)
    buffer[tx] += input_k[i]*(float)(i+1);
  
  // add (reduce)
  for (unsigned int stride = WINDOWGATE2_THREADS >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  
  if (tx == 0)
  {
    float centroid = buffer[0];
    
    // make centroid a number between 0 and 1
    centroid /= (float)(inputSize);
    
    normalizedCentroids[k] = centroid;
    if ( train )
    {
      centroid += noise[k];
      centroid = fminf(fmaxf(0,centroid),1);
    }
    // align centroid to output
    centroid *= (float)(outputSize);
    
    float inputIdx = centroid/(float)(inputSize) - 0.5*(float)inputWindowSize;
    float outputIdx = centroid - 0.5*(float)outputWindowSize;
    
    // clip indices
    inputIdx = fminf(inputIdx, inputSize-inputWindowSize+1);
    inputIdx = fmaxf(inputIdx, 1);
    outputIdx = fminf(outputIdx, outputSize-outputWindowSize+1);
    outputIdx = fmaxf(outputIdx, 1);
    
    inputIdx = ceilf(inputIdx);
    outputIdx = ceilf(outputIdx);
    // align centroid to outputWindow
    centroid -= (outputIdx-1);
    
    inputIndice[k] = (int)inputIdx;
    outputIndice[k] = (int)outputIdx;
    centroids[k] = centroid;
    
    buffer[WINDOWGATE2_THREADS] = inputIdx;
  }
  
  __syncthreads();
  
  float inputIdx = buffer[WINDOWGATE2_THREADS];
  const float *inputWindow = input_k + (int)inputIdx;
  
  for (int i=tx; i<outputWindowSize; i+=blockDim.x)
  {
    output_k[i] = inputWindow[(int)floorf(((float)i)/windowStride)];
  }
}
  
static int cunnx_WindowGate2_updateOutput(lua_State *L)
{ 
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int inputWindowSize = luaT_getfieldcheckint(L, 1, "inputWindowSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
  int windowStride = luaT_getfieldcheckint(L, 1, "windowStride");
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int train = luaT_getfieldcheckboolean(L, 1, "train");
  
  THCudaLongTensor *outputIndiceCuda = (THCudaLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceCuda", "torch.CudaLongTensor");
  THCudaTensor *inputIndiceCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "inputIndiceCuda", "torch.CudaTensor");
  THLongTensor *outputIndice = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndice", "torch.LongTensor");
  THCudaTensor *centroid = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "centroid", "torch.CudaTensor");
  THCudaTensor *normalizedCentroid = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "normalizedCentroid", "torch.CudaTensor");
  THCudaTensor *noise = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "noise", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_output", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  
  THCudaTensor_resize2d(state, output, batchSize, outputWindowSize);
  THCudaLongTensor_resize1d(state, outputIndiceCuda, batchSize);
  THLongTensor_resize1d(outputIndice, batchSize);
  THCudaTensor_resize1d(state, inputIndiceCuda, batchSize);
  THCudaTensor_resize1d(state, centroid, batchSize);
  THCudaTensor_resize1d(state, normalizedCentroid, batchSize);
  
  
  /* call cudakernel */
  dim3 blocks(batchSize); // each cuda-block is an example
  dim3 threads(WINDOWGATE2_THREADS);
  cunnx_WindowGate2_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, output), THCudaTensor_data(state, centroid),
    THCudaTensor_data(state, normalizedCentroid), THCudaTensor_data(state, inputIndiceCuda),
    (float *)THCudaLongTensor_data(state, outputIndiceCuda),
    (const float*)THCudaTensor_data(state, input), (const float*)THCudaTensor_data(state, noise), 
    inputSize, outputSize, inputWindowSize, outputWindowSize, windowStride, train
  );
  
  THLongTensor_copyCuda(state, outputIndice, outputIndiceCuda);
  
  return 0;
}

__global__ void cunnx_WindowGate2_updateGradInput_kernel(
  float *gradInput, float *error, float* targetCentroids, 
  const float *centroids,const float *input,
  const float *inputIndice, const float *outputIndice,
  const float* output, const float* gradOutput, 
  int inputSize, int outputSize, int inputWindowSize, 
  int outputWindowSize, int windowStride, float c, float d, float e, float lr)
{
  unsigned int tx = threadIdx.x;
  unsigned int k = blockIdx.x;
  const float *gradOutput_k = gradOutput + outputWindowSize*k;
  float *gradInput_k = gradInput + inputSize*k;

  
  float *gradInputWindow = gradInput_k + (int)(inputIndice[k] - 1);
  
  for (int i=tx; i<inputWindowSize; i+=blockDim.x)
  {
    float sum = 0;
    const float *gradOutputChannel = gradOutput_k + i*windowStride;
    for (int j=0; j<windowStride; j++)
      sum += gradOutputChannel[j];
      
    gradInputWindow[i] += sum; 
  }
  
}

  
static int cunnx_WindowGate2_updateGradInput(lua_State *L)
{ 
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor"); 
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int inputWindowSize = luaT_getfieldcheckint(L, 1, "inputWindowSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
  int windowStride = luaT_getfieldcheckint(L, 1, "windowStride");
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  
  float c = (float)luaT_getfieldchecknumber(L, 1, "c");
  float d = (float)luaT_getfieldchecknumber(L, 1, "d");
  float e = (float)luaT_getfieldchecknumber(L, 1, "e");
  float lr = (float)luaT_getfieldchecknumber(L, 1, "lr");
  
  THCudaTensor *error = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "error", "torch.CudaTensor");
  THCudaTensor *centroid = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "centroid", "torch.CudaTensor");
  THCudaTensor *targetCentroid = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "targetCentroid", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_output", "torch.CudaTensor");
  THCudaTensor *outputIndiceCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceCuda", "torch.CudaTensor");
  THCudaTensor *inputIndiceCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "inputIndiceCuda", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  
  THCudaTensor_resize2d(state, gradInput, batchSize, inputSize);
  THCudaTensor_fill(state, gradInput, 0);
  THCudaTensor_resize1d(state, error, batchSize);
  THCudaTensor_resize1d(state, targetCentroid, batchSize);
    
  /* call cudakernel */
  dim3 blocks(batchSize); // each cuda-block is an example
  dim3 threads(WINDOWGATE2_THREADS);
  cunnx_WindowGate2_updateGradInput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, gradInput), THCudaTensor_data(state, error), 
    THCudaTensor_data(state, targetCentroid), 
    (const float*)THCudaTensor_data(state, centroid),
    (const float*)THCudaTensor_data(state, input), 
    (const float*)THCudaTensor_data(state, inputIndiceCuda),
    (const float*)THCudaTensor_data(state, outputIndiceCuda),
    (const float*)THCudaTensor_data(state, output), 
    (const float*)THCudaTensor_data(state, gradOutput), 
    inputSize, outputSize, inputWindowSize, outputWindowSize, 
    windowStride, c, d, e, lr
  );
  
  return 1;
}


static const struct luaL_Reg cunnx_WindowGate2__ [] = {
  {"WindowGate2_updateOutput", cunnx_WindowGate2_updateOutput},
  {"WindowGate2_updateGradInput", cunnx_WindowGate2_updateGradInput},
  {NULL, NULL}
};

static void cunnx_WindowGate2_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_WindowGate2__, "nn");
  lua_pop(L,1);
}
