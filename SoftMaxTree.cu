#include "utils.h"
#define SOFTMAXTREE_THREADS 32
#define SOFTMAXTREE_MAXCHILDREN 10000

__global__ void cunnx_SoftMaxTree_updateOutput_kernel(
  float *output, float *logsoftOutput, float *input, float *weight, 
  float *bias, float *target, float *childParent, float *parentChildren, 
  int nInput, int rootId, int maxFamilyPath)
{
  __shared__ float buffer[SOFTMAXTREE_THREADS+1];
  __shared__ float linearOutput[SOFTMAXTREE_MAXCHILDREN];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  float *input_k = input + k*nInput;
  float *nodeOutput, *nodeWeight, *nodeBias;
  float narrowsum = 0;
  int childId = target[k] - 1;
  int parentId, parentIdx, childIdx, nChildren;
  float *node;
  int n = 0;

  // loop through nodes
  while(1)
  {
    /* get next Node in Tree */
    node = childParent + childId*2;
    parentId = (int)node[0] - 1;
    childIdx = (int)node[1] - 1;
    
    node = parentChildren + parentId*2;
    parentIdx = (int)node[0] - 1;
    nChildren = (int)node[1];
    
    CudaAssert(childIdx < nChildren)
    /* Linear */
    
    nodeWeight = weight + parentIdx*nInput;
    nodeBias = bias + parentIdx;
    
    // addmv (dot products)
    for (int j=0; j<nChildren; j++)
    {
      // zero buffer
      buffer[tx] = 0;
      
      // multiply
      for (int i=tx; i<nInput; i+=i_step)
      {
        buffer[tx] += input_k[i]*nodeWeight[j*nInput + i];
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
        linearOutput[j] = buffer[0] + nodeBias[j];
      }
    }
    
    __syncthreads();
    
    /* LogSoftMax */
    nodeOutput = logsoftOutput + maxFamilyPath*k + n;
    
    // max?
    buffer[tx] = -FLT_MAX;
    for (int i=tx; i<nChildren; i+=i_step)
    {
      float z = linearOutput[i];
      if(buffer[tx] < z)
        buffer[tx] = z;
    }
    
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
      __syncthreads();
      if ((tx < stride) && (buffer[tx] < buffer[tx+stride]))
        buffer[tx] = buffer[tx+stride];
    }
    if (tx == 0)
    {
      float max_k = -FLT_MAX;
      if(max_k < buffer[0])
        max_k = buffer[0];
      buffer[SOFTMAXTREE_THREADS] = max_k;
    }

    __syncthreads();
    
    // logadd?
    float max_k = buffer[SOFTMAXTREE_THREADS];
    buffer[tx] = 0;
    for (int i=tx; i<nChildren; i+=i_step)
    {
      buffer[tx] += expf(linearOutput[i]-max_k);
    }

    // reduce
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
      __syncthreads();
      if (tx < stride)
        buffer[tx] += buffer[tx+stride];
    }
    if (tx == 0)
    {
      float m = max_k + logf(buffer[0]);
      buffer[SOFTMAXTREE_THREADS] = m;
    }

    __syncthreads();

    // logsoftmax
    float logsum_k = buffer[SOFTMAXTREE_THREADS];
    for (int i=tx; i<nChildren; i+=i_step)
    {
      nodeOutput[i] = linearOutput[i] - logsum_k;
    }
      
    __syncthreads();
    
    /* Narrow + CAddTable (without log, would have been CMulTable) */
    if (tx == 0)
      narrowsum += nodeOutput[childIdx];
      
    n += nChildren;
    CudaAssert((n <= maxFamilyPath))
    /* Break when root is reached */
    if (parentId == rootId) 
    {
      break;
    }
    childId = parentId;
  }
  if (tx == 0) 
  {
    output[k] = narrowsum;
  }
}


static int cunnx_SoftMaxTree_updateOutput(lua_State *L)
{ 
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamilyPath = (int)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  int maxFamily = (int)luaT_getfieldcheckint(L, 1, "maxFamily");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParentCuda", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildrenCuda", "torch.CudaTensor");

  THCudaTensor *logsoftOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");  
  luaL_argcheck(L, maxFamily <= SOFTMAXTREE_MAXCHILDREN, 2, "Hierarchy has node(s) with too many children");
  
  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resize1d(state, output, input->size[0]);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, output), THCudaTensor_data(state, logsoftOutput), 
    THCudaTensor_data(state, input), THCudaTensor_data(state, weight), 
    THCudaTensor_data(state, bias), THCudaTensor_data(state, target), 
    THCudaTensor_data(state, childParent), THCudaTensor_data(state, parentChildren), 
    input->size[1], rootId, maxFamilyPath
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  THCudaTensor_free(state, input);
  return 1;
}


__global__ void cunnx_SoftMaxTree_updateGradInput_kernel(
  float *gradInput, float *logsoftOutput, float *gradOutput, float* weight,
  float *target, float *childParent, float *parentChildren, 
  int nInput, int rootId, int maxFamilyPath)
{
  __shared__ float buffer[SOFTMAXTREE_THREADS];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  float *gradInput_k = gradInput + k*nInput;
  float *nodeGrad, *nodeWeight;
  float grad = gradOutput[k];
  int childId = target[k] - 1;
  int parentId, parentIdx, childIdx, nChildren;
  float *node;
  int n = 0;
  
  // zero gradInputs (for accumulation)
  for (int i=tx; i<nInput; i+=i_step)
    gradInput_k[i] = 0;

  // loop through nodes
  while(1)
  {
    /* get next Node in Tree */
    node = childParent + childId*2;
    parentId = (int)node[0] - 1;
    childIdx = (int)node[1] - 1;
    
    node = parentChildren + parentId*2;
    parentIdx = (int)node[0] - 1;
    nChildren = (int)node[1];
    
    /* CAddTable + Narrow + LogSoftMax */
    // AKA linearGradOutput (we reuse the _multiBuffer Tensor)
    nodeGrad = logsoftOutput + maxFamilyPath*k + n; 

    for(int i=tx; i<nChildren; i+=i_step)
    {
      nodeGrad[i] = -expf(nodeGrad[i])*grad;
    }
    
    __syncthreads();
    if (tx == 0)
    {
      nodeGrad[childIdx] += grad;
    }
      
    __syncthreads();

    /* Linear */
    nodeWeight = weight + parentIdx*nInput;
    
    // addmv (dot products)
    for (int i=tx; i<nInput; i+=i_step)
    {
     // zero buffer
      buffer[tx] = 0;
      
      for (int j=0; j<nChildren; j++)
      {
        // multiply
        buffer[tx] += nodeGrad[j]*nodeWeight[j*nInput + i];
      }
      // accumulate into global memory
      gradInput_k[i] += buffer[tx];
    }
    
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

static int cunnx_SoftMaxTree_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamilyPath = (int)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParentCuda", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildrenCuda", "torch.CudaTensor");
  
  THCudaTensor *logsoftOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_gradInput", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");  
  
  luaL_argcheck(L, gradOutput->nDimension == 1, 2, "1D tensor expected");
  
  THCudaTensor_resizeAs(state, gradInput, input);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_updateGradInput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, gradInput), THCudaTensor_data(state, logsoftOutput), 
    THCudaTensor_data(state, gradOutput), THCudaTensor_data(state, weight), 
    THCudaTensor_data(state, target), THCudaTensor_data(state, childParent), 
    THCudaTensor_data(state, parentChildren), 
    input->size[1], rootId, maxFamilyPath
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  return 1;
}

__global__ void cunnx_SoftMaxTree_accGradParameters_kernel(
  float *gradWeight, float *gradBias, float *input, 
  float *linearGradOutput, int *nodeUpdateCuda, float *target, 
  float *childParent, float *parentChildren, 
  int nInput, int rootId, int maxFamilyPath, int maxDept, float scale)
{
  __shared__ float buffer[SOFTMAXTREE_THREADS];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  float *input_k = input + k*nInput;
  float *nodeGradOutput, *nodeGradWeight, *nodeGradBias;
  // reuse _multiBuffer for keeping track of which node gets gradients
  int *nodeUpdate = nodeUpdateCuda + maxDept*k; 
  int childId = target[k] - 1;
  int parentId, parentIdx, nChildren;
  float *node;
  int n = 0;
  int m = 0;
  
  // loop through nodes
  while(1)
  {
    /* get next Node in Tree */
    node = childParent + childId*2;
    parentId = (int)node[0] - 1;
    
    node = parentChildren + parentId*2;
    parentIdx = (int)node[0] - 1;
    nChildren = (int)node[1];
    
    nodeGradOutput = linearGradOutput + maxFamilyPath*k + n; 
    nodeGradWeight = gradWeight + parentIdx*nInput;
    nodeGradBias = gradBias + parentIdx;
    
    // addr weights (scalar-products)
    for (int i=tx; i<nInput; i+=i_step)
    {
      // copy input to buffer
      buffer[tx] = input_k[i]; // replace shared with register?
    
      for (int j=0; j<nChildren; j++)
      {
        // multiply accumulate weights
        float dw = scale*nodeGradOutput[j]*buffer[tx];
        atomicAdd(&nodeGradWeight[j*nInput + i], dw);
      }
    }
    
    // cadd bias
    for (int j=tx; j<nChildren; j+=i_step)
    {
      // multiply accumulate biases
      float db = scale*nodeGradOutput[j];
      atomicAdd(&nodeGradBias[j], db);
    }
    
    // keep track of which node gets gradients
    nodeUpdate[m] = (int)parentId;
    
    n += nChildren;
    CudaAssert((n <= maxFamilyPath))
    m += 1;
    CudaAssert((m <= maxDept))
    /* Break when root is reached */
    if (parentId == rootId)
    {
      if (m < maxDept)
        nodeUpdate[m] = -1; // zero means end of buffer
      break;
    }
    childId = parentId;
  }
}

static int cunnx_SoftMaxTree_accGradParameters(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");  
  float scale = luaL_optnumber(L, 5, 1);
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamilyPath = (int)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  int maxDept = (int)luaT_getfieldcheckint(L, 1, "maxDept");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParentCuda", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildrenCuda", "torch.CudaTensor");
  
  THCudaTensor *linearGradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  THCudaIntTensor *nodeUpdateCuda = (THCudaIntTensor*)luaT_getfieldcheckudata(L, 1, "_nodeUpdateCuda", "torch.CudaIntTensor");
  THIntTensor *nodeUpdateHost = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "_nodeUpdateHost", "torch.IntTensor");
  
  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  
  int i, j;
  THIntTensor *nodeUpdate;
  
  lua_getfield(L, 1, "updates");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  
  input = THCudaTensor_newContiguous(state, input); 
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_accGradParameters_kernel<<<blocks,threads>>>(
    THCudaTensor_data(state, gradWeight), THCudaTensor_data(state, gradBias), 
    THCudaTensor_data(state, input), THCudaTensor_data(state, linearGradOutput), 
    THCudaIntTensor_data(state, nodeUpdateCuda), THCudaTensor_data(state, target), 
    THCudaTensor_data(state, childParent), THCudaTensor_data(state, parentChildren), 
    input->size[1], rootId, maxFamilyPath, maxDept, scale 
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  // copy updated nodeIds from device to host
  THIntTensor_copyCuda(state, nodeUpdateHost, nodeUpdateCuda);
  nodeUpdate = THIntTensor_new();
  
  // fill updates table
  for (i=0; i<nodeUpdateHost->size[0]; i++)
  {
    THIntTensor_select(nodeUpdate, nodeUpdateHost, 0, i);
    
    for (j=0; j<nodeUpdateHost->size[1]; j++)
    {
      int nodeId = THIntTensor_get1d(nodeUpdate, j);
      double count;
      
      if (nodeId == -1)
      {
        break;
      }
      
      /* updates will contain nodeId (key) sum of scales (value)*/
      lua_pushinteger(L, (int)(nodeId+1));
      lua_gettable(L, -2);
      count = lua_tonumber(L, -1) + scale;
      lua_pop(L, 1);
      
      lua_pushinteger(L, (int)(nodeId+1)); /* key */
      lua_pushnumber(L, count); /* value */
      lua_settable(L, -3);
    }
  }
  
  THIntTensor_free(nodeUpdate);
  return 0;
}

static const struct luaL_Reg cunnx_SoftMaxTree__ [] = {
  {"SoftMaxTree_updateOutput", cunnx_SoftMaxTree_updateOutput},
  {"SoftMaxTree_updateGradInput", cunnx_SoftMaxTree_updateGradInput},
  {"SoftMaxTree_accGradParameters", cunnx_SoftMaxTree_accGradParameters},
  {NULL, NULL}
};

static void cunnx_SoftMaxTree_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_SoftMaxTree__, "nn");
  lua_pop(L,1);
}
