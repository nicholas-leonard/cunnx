#include "utils.h"
#define WINDOWGATE_THREADS 128

__global__ void cunnx_WindowGate_updateOutput_kernel(
  float *output, float *centroids, float *normalizedCentroids, float *outputIndice,
  const float *input, const float *noise, int inputSize, int outputSize, 
  int outputWindowSize, float a, float b, int train)
{
  __shared__ float buffer[WINDOWGATE_THREADS];
  unsigned int tx = threadIdx.x;
  unsigned int k = blockIdx.x;
  const float *input_k = input + inputSize*k;
  float *output_k = output + outputWindowSize*k;
  
  // get coordinate of centoid
  buffer[tx] = 0;
  for (unsigned int i=tx; i<inputSize; i+=blockDim.x)
    buffer[tx] += input_k[i]*(float)(i+1);
  
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
    
    float outputIdx = centroid - 0.5*(float)outputWindowSize;
    
    // clip indices
    outputIdx = fminf(outputIdx, outputSize-outputWindowSize+1);
    outputIdx = fmaxf(outputIdx, 1);
    
    outputIdx = ceilf(outputIdx);
    // align centroid to outputWindow
    centroid -= (outputIdx-1);
    
    outputIndice[k] = (int)outputIdx;
    centroids[k] = centroid;
    buffer[0] = centroid;
  }
  
  __syncthreads();
  
  float centroid = buffer[0];
   
  // gaussian blur 
  for (int i=tx; i<outputWindowSize; i+=blockDim.x)
  {
    float x = (float)(i+1)-centroid;
    output_k[i] = a*expf(x*x*b);
  }
}
  
static int cunnx_WindowGate_updateOutput(lua_State *L)
{ 
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int train = luaT_getfieldcheckboolean(L, 1, "train");
  float a = (float)luaT_getfieldchecknumber(L, 1, "a");
  float b = (float)luaT_getfieldchecknumber(L, 1, "b");
  
  THCudaLongTensor *outputIndiceCuda = (THCudaLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceCuda", "torch.CudaLongTensor");
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
  THCudaTensor_resize1d(state, centroid, batchSize);
  THCudaTensor_resize1d(state, normalizedCentroid, batchSize);
  
  /* call cudakernel */
  dim3 blocks(batchSize); // each cuda-block is an example
  dim3 threads(WINDOWGATE_THREADS);
  cunnx_WindowGate_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, output), THCudaTensor_data(state, centroid),
    THCudaTensor_data(state, normalizedCentroid), (float *)THCudaLongTensor_data(state, outputIndiceCuda),
    (const float*)THCudaTensor_data(state, input), (const float*)THCudaTensor_data(state, noise), 
    inputSize, outputSize, outputWindowSize, a, b, train
  );
  
  THLongTensor_copyCuda(state, outputIndice, outputIndiceCuda);
  
  return 0;
}

__global__ void cunnx_WindowGate_updateGradInput_kernel(
  float *gradInput, float *error, float* targetCentroids, 
  const float *centroids,const float *input, const float *outputIndice,
  const float* output, const float* gradOutput, 
  int inputSize, int outputSize, int outputWindowSize,
  float c, float d, float e, float lr)
{
  __shared__ float buffer[WINDOWGATE_THREADS+1];
  unsigned int tx = threadIdx.x;
  unsigned int k = blockIdx.x;
  const float *gradOutput_k = gradOutput + outputWindowSize*k;
  const float *output_k = output + outputWindowSize*k;
  const float *input_k = input + inputSize*k;
  float *gradInput_k = gradInput + inputSize*k;
  float centroid = centroids[k];

  // get gradient of centroid
  buffer[tx] = 0;
  for (unsigned int i=tx; i<outputWindowSize; i+=blockDim.x)
  {
    buffer[tx] += gradOutput_k[i]*output_k[i]*((float)(i+1) - centroid);
  }
  
  // add (reduce)
  for (unsigned int stride = WINDOWGATE_THREADS >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  
  if (tx == 0)
  {
    int outputIdx = outputIndice[k];
    float gradCentroid = buffer[0]*c;
    centroid -= (lr*gradCentroid);
    centroid += outputIdx-1;
    centroid /= (float)(outputSize);
    targetCentroids[k] = centroid;
    buffer[WINDOWGATE_THREADS] = centroid*(float)(inputSize);
  }
  
  __syncthreads();
  float targetCentroid = buffer[WINDOWGATE_THREADS];
   
  buffer[tx] = 0;
  // target is a gaussian blur 
  for (int i=tx; i<inputSize; i+=blockDim.x)
  {
    float target = (float)(i+1)-targetCentroid;
    target = d*expf(target*target*e);
    float input = input_k[i];
    // dot product of logProbInput and probTarget (NLL)
    buffer[tx] -= logf(input + 0.0000001)*target;
    // grad input w.r.t. NLL
    gradInput_k[i] = -target/(input + 0.0000001);
  }
  
  // add (reduce)
  for (unsigned int stride = WINDOWGATE_THREADS >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  
  if (tx == 0)
    error[k] = buffer[tx];
}

  
static int cunnx_WindowGate_updateGradInput(lua_State *L)
{ 
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor"); 
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
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
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  
  THCudaTensor_resize2d(state, gradInput, batchSize, inputSize);
  THCudaTensor_resize1d(state, error, batchSize);
  THCudaTensor_resize1d(state, targetCentroid, batchSize);
    
  /* call cudakernel */
  dim3 blocks(batchSize); // each cuda-block is an example
  dim3 threads(WINDOWGATE_THREADS);
  cunnx_WindowGate_updateGradInput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, gradInput), THCudaTensor_data(state, error), 
    THCudaTensor_data(state, targetCentroid), 
    (const float*)THCudaTensor_data(state, centroid),
    (const float*)THCudaTensor_data(state, input), 
    (const float*)THCudaTensor_data(state, outputIndiceCuda),
    (const float*)THCudaTensor_data(state, output), 
    (const float*)THCudaTensor_data(state, gradOutput), 
    inputSize, outputSize, outputWindowSize, c, d, e, lr
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
