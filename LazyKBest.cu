#include "utils.h"
#define LAZYKBEST_THREADS 32

__global__ void cunnx_LazyKBest_updateOutput_kernel(
  float *output, float *indice, const float *input, 
  int inputSize, int outputSize)
{
  __shared__ float bufferVal[LAZYKBEST_THREADS];
  __shared__ float bufferIdx[LAZYKBEST_THREADS];
  const int tx = threadIdx.x;
  const int step = blockDim.x;
  const int k = blockIdx.x;
  
  float *output_k = output + k*outputSize;
  float *indice_k = indice + k*outputSize;
  const float *input_k = input + k*inputSize;
  
  float maxVal = -FLT_MAX;
  int maxIdx = -1;
  
  for (int i=tx; i<inputSize; i+=step)
  {
    float val = input_k[i];
    if (val > maxVal)
    {
      maxVal = val;
      maxIdx = i;
    }
  }
  
  bufferVal[tx] = maxVal;
  bufferIdx[tx] = maxIdx;
  
  // reduce
  for (unsigned int stride = blockDim.x >> 1; stride > outputSize-1; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
    {
      float val = bufferVal[tx+stride];
      if (val > bufferVal[tx])
      {
        bufferVal[tx] = val;
        bufferIdx[tx] = bufferIdx[tx+stride];
      }
    }
  }
  
  if (tx < outputSize)
  {
    output_k[tx] = bufferVal[tx];
    indice_k[tx] = bufferIdx[tx] + 1;
  }
}


static int cunnx_LazyKBest_updateOutput(lua_State *L)
{   
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_output", "torch.CudaTensor");
  THCudaTensor *indice = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_indice", "torch.CudaTensor");
  int k = luaT_getfieldcheckint(L, 1, "k");

  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, k <= LAZYKBEST_THREADS, 1, "k must be smaller than KBEST_THREADS");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, input), 2, "Expecting contiguous input");
  
  THCudaTensor_resize2d(state, output, input->size[0], k);
  THCudaTensor_resize2d(state, indice, input->size[0], k);
 
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(LAZYKBEST_THREADS);
  cunnx_LazyKBest_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, output), THCudaTensor_data(state, indice), 
    THCudaTensor_data(state, input), input->size[1], k
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
}
 
 
__global__ void cunnx_LazyKBest_updateGradInput_kernel(
  float *gradInput, const float *indice, const float *gradOutput, 
  int inputSize, int outputSize)
{
  int tx = threadIdx.x;
  int step = blockDim.x;
  int k = blockIdx.x;
  
  float *gradInput_k = gradInput + k*inputSize;
  const float *gradOutput_k = gradOutput + k*outputSize;
  const float *indice_k = indice + k*outputSize;
  
  for (int i=tx; i<outputSize; i+=step)
    gradInput_k[(int)(indice_k[i] - 1)] = gradOutput_k[i];
}


static int cunnx_LazyKBest_updateGradInput(lua_State *L)
{   
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *indice = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  int k = luaT_getfieldcheckint(L, 1, "k");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, indice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, input), 2, "Expecting contiguous input");
  
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_fill(state, gradInput, 0);
 
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(LAZYKBEST_THREADS);
  cunnx_LazyKBest_updateGradInput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, gradInput), THCudaTensor_data(state, indice), 
    THCudaTensor_data(state, gradOutput), input->size[1], k
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
} 
  
static const struct luaL_Reg cunnx_LazyKBest__ [] = {
  {"LazyKBest_updateOutput", cunnx_LazyKBest_updateOutput},
  {"LazyKBest_updateGradInput", cunnx_LazyKBest_updateGradInput},
  {NULL, NULL}
};

static void cunnx_LazyKBest_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_LazyKBest__, "nn");
  lua_pop(L,1);
}
