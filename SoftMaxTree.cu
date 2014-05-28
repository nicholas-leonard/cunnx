#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAXTREE_THREADS 32
#define SOFTMAXTREE_MAXCHILDREN 10000


__global__ void cunnx_SoftMaxTree_updateOutput_kernel(
  float *output, float* logsoftOutput, 
  float *input, float* weight, float* bias, 
  int* target, int* childParent, int* parentChildren, 
  int nInput, int rootId)
{
  //__shared__ float input_buffer[nInput]; // constant might be faster
  __shared__ float buffer[SOFTMAXTREE_THREADS+1];
  __shared__ float linearOutput[SOFTMAXTREE_MAXCHILDREN];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  float *input_k = input + k*nInput;
  float *nodeOutput, *nodeWeight, *nodeBias;
  float narrowsum = 0;
  int childId = (*(target+k)) - 1;
  int parentId, parentIdx, childIdx, nChildren;
  int nOutput;
  int *node;
  int n = 0;
  
  // zero buffer
  buffer[tx] = 0;
  
  __syncthreads();

  // loop through nodes
  while(1)
  {
    /* get next Node in Tree */
    node = childParent + childId*2;
    parentId = (*node) - 1;
    childIdx = (*(node+1)) - 1;
    
    node = parentChildren + parentId*2;
    parentIdx = (*node) - 1;
    nChildren = *(node+1);
    
    /* Linear */
    
    nodeWeight = weight + parentIdx*nInput;
    nodeBias = bias + parentIdx;
    
    // addmv (dot products)
    for (int j=0; j<nChildren; j++)
    {
      // multiply
      for (int i=tx; i<nInput; i+=i_step)
      {
        buffer[tx] += input_k[i]*nodeWeight[i*nInput + j];
      }
      // add (reduce)
      for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
      {
        __syncthreads();
        if (tx < stride)
          buffer[tx] += buffer[tx+stride];
      }
      if (tx == 0) 
        linearOutput[j] = buffer[0] + nodeBias[j];
    }
    
    __syncthreads();
    
    /* LogSoftMax */
    nodeOutput = logsoftOutput + n;
    
    // max?
    buffer[tx] = -FLT_MAX;
    for (int i=tx; i<nChildren; i+=i_step)
    {
      float z = linearOutput[i];
      if(buffer[tx] < z)
        buffer[tx] = z;
    }

    __syncthreads();
    
    
    // reduce
    nOutput = blockDim.x;
    if (nChildren < nOutput)
      nOutput = nChildren;
    if (tx == 0)
    {
      float max_k = -FLT_MAX;
      for (int i=0; i<nOutput; i++)
      {
        if(max_k < buffer[i])
          max_k = buffer[i];
      }
      buffer[SOFTMAXTREE_THREADS] = max_k;
    }

    __syncthreads();

    // logadd?
    float max_k = buffer[SOFTMAXTREE_THREADS];
    buffer[tx] = 0;
    for (int i=tx; i<nOutput; i+=i_step)
      buffer[tx] += __expf(linearOutput[i]-max_k);

    __syncthreads();

    // reduce
    if (tx == 0)
    {
      float logsum_k = 0;
      for (int i=0; i<nOutput; i++)
        logsum_k += buffer[i];
      buffer[SOFTMAXTREE_THREADS] = max_k + __logf(logsum_k);
    }

    __syncthreads();

    // logsoftmax
    float logsum_k = buffer[SOFTMAXTREE_THREADS];
    for (int i=tx; i<nOutput; i+=i_step)
      nodeOutput[i] = linearOutput[i] - logsum_k;
      
    /* Narrow + CAddTable (without log, would have been CMulTable) */
    narrowsum += nodeOutput[childIdx];
    
    n += nChildren;
    /* Break when root is reached */
    if (parentId == rootId) 
    {
      break;
    }
    childId = parentId;
  }
  output[k] = narrowsum;
}

static int cunnx_SoftMaxTree_updateOutput(lua_State *L)
{ 
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamily = (int)luaT_getfieldcheckint(L, 1, "maxFamily");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.CudaTensor");
  
  THCudaTensor *linearOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_nodeBuffer", "torch.CudaTensor");
  THCudaTensor *logsoftOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");  
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(logsoftOutput), 
    THCudaTensor_data(input), THCudaTensor_data(weight), 
    THCudaTensor_data(bias), (int*)THCudaTensor_data(target), 
    (int*)THCudaTensor_data(childParent), (int*)THCudaTensor_data(parentChildren), 
    input->size[1], rootId
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(input);
  return 1;
}

static int cunnx_SoftMaxTree_updateGradInput(lua_State *L)
{
  return 1;
}

static const struct luaL_Reg cunnx_SoftMaxTree__ [] = {
  {"SoftMaxTree_updateOutput", cunnx_SoftMaxTree_updateOutput},
  {"SoftMaxTree_updateGradInput", cunnx_SoftMaxTree_updateGradInput},
  {NULL, NULL}
};

static void cunnx_SoftMaxTree_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_SoftMaxTree__, "nn");
  lua_pop(L,1);
}
