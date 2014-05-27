#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAXTREE_THREADS 32


__global__ void cunnx_SoftMaxTree_updateOutput_kernel(
  float *output, float* linearoutput, float* logsoftoutput, 
  float *input, float* weight, float* bias, 
  int* target, int* childParent, int* parentChildren, 
  int nInput, int rootId)
{
  //__shared__ float input_buffer[nInput]; // constant might be faster
  __shared__ float buffer[SOFTMAXTREE_THREADS+1];
  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  float *input_k = input + k*nInput;
  float narrowsum = 0;
  int childId = (*(target+k)) - 1;
  int parentId, parentIdx, childIdx, nChildren;
  int nOutput;
  int *node;
  int n = 0;
  
  // zero buffer
  buffer[i] = 0;
  
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
    nodeOutput = linearOutput + n;
    
    // addmv (dot products)
    for (int j=0; j<nChildren; j++)
    {
      for (int i=i_start; i<nInput; i+=i_step)
      {
        buffer[i_start] += input_k[i]*nodeWeight[i*nInput + j];
      }
      for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
      {
        __syncthreads();
        if (i_start < stride)
          buffer[i_start] += buffer[i_start+stride];
      }
      if (i_start == 0) 
        *nodeOutput = buffer[0] + nodeBias[j];
    }
    
    __syncthreads();
    
    /* LogSoftMax */
    nodeInter = nodeOutput;
    nodeOutput = logsoftOutput + n;
    
    // max?
    buffer[i_start] = -FLT_MAX;
    for (int i=i_start; i<nChildren; i+=i_step)
    {
      float z = nodeInder[i];
      if(buffer[i_start] < z)
        buffer[i_start] = z;
    }

    __syncthreads();
    
    
    // reduce
    nOutput = blockDim.x
    if (nChildren < nOutput)
      nOutput = nChildren;
    if (i_start == 0)
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
    buffer[i_start] = 0;
    for (int i=i_start; i<nOutput; i+=i_step)
      buffer[i_start] += __expf(nodeInter[i]-max_k);

    __syncthreads();

    // reduce
    if (i_start == 0)
    {
      float logsum_k = 0;
      for (int i=0; i<nOutput; i++)
        logsum_k += buffer[i];
      buffer[SOFTMAXTREE_THREADS] = max_k + __logf(logsum_k);
    }

    __syncthreads();

    // logsoftmax
    float logsum_k = buffer[SOFTMAXTREE_THREADS];
    for (int i=i_start; i<nOutput; i+=i_step)
      nodeOutput[i] = nodeInter[i] - logsum_k;
      
    /* Narrow + CAddTable (without log, would have been CMulTable) */
    narrowsum += nodeOutput[childIdx]
    
    n += nChildren;
    /* Break when root is reached */
    if (parentId == rootId) 
    {
      break;
    }
    childId = parentId;
  }
  *output_k = narrowsum;
}

static int cunnx_SoftMaxTree_updateOutput(lua_State *L)
{ 
  THCudaTensor *input = luaT_checkudata(L, 2, "torch.CudaTensor");  
  THIntTensor *target = (THIntTensor*)luaT_checkudata(L, 3, "torch.IntTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId") - 1);
  
  THIntTensor *childParent = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.IntTensor");
  THIntTensor *parentChildren = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.IntTensor");
  
  THCudaTensor *linearOutput = luaT_getfieldcheckudata(L, 1, "_linearOutput", "torch.CudaTensor");
  THCudaTensor *logsoftOutput = luaT_getfieldcheckudata(L, 1, "_logsoftmaxOutput", "torch.CudaTensor");
  
  THCudaTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  THIntTensor *node;

  long i, d;
  long n, m = 0;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");

  node = THIntTensor_new();
  
  THCudaTensor_resize1d(output, input->size[0]);
  
  /* Get sum of nodeChildren and number of nodes */
  for(i = 0; i < input->size[0]; i++)
  {
    long childId = (long)(THIntTensor_get1d(target, i)) - 1;
    while(1)
    {
      long parentId, nChildren;
      /* get next Node in Tree */
      THIntTensor_select(node, childParent, 0, childId);
      parentId = (long)(THIntTensor_get1d(node, 0)) - 1;
      
      luaL_argcheck(L, parentId != -2, 2, "Non-root node has no parent in tree.");
      
      THIntTensor_select(node, parentChildren, 0, parentId);
      nChildren = (long)(THIntTensor_get1d(node, 1));
      
      n += nChildren;
      m += 1;
      /* Break when root is reached */
      if (parentId == rootId) 
      {
        break;
      }
      childId = parentId;
    }
  }
  THIntTensor_free(node);
  
  // we use these to keep intermediate results for later backprop
  THCudaTensor_resize2d(linearOutput, n);
  THCudaTensor_resize1d(logsoftOutput, n);
  
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(linearOutput), 
    THCudaTensor_data(logsoftOutput), THCudaTensor_data(input), 
    THCudaTensor_data(weight), THCudaTensor_data(bias), 
    THCudaTensor_data(target), THCudaTensor_data(childParent), 
    THCudaTensor_data(parentChildren), input->size[1], root_id
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

static const struct luaL_Reg cunnx_SoftMax__ [] = {
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
