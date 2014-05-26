#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAXTREE_THREADS 128

struct addvalue_functor
{
  const float value;

  addvalue_functor(float value_) : value(value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x+value);
  }
};

__global__ void cunn_LogSoftMax_updateOutput_kernel(float *output, float *input, int nframe, int dim)
{
  __shared__ float buffer[LOGSOFTMAX_THREADS+1];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i];
    if(buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float max_k = -FLT_MAX;
    for (int i=0; i<blockDim.x; i++)
    {
      if(max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[LOGSOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // logadd?
  float max_k = buffer[LOGSOFTMAX_THREADS];
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += __expf(input_k[i]-max_k);

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float logsum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      logsum_k += buffer[i];
    buffer[LOGSOFTMAX_THREADS] = max_k + __logf(logsum_k);
  }

  __syncthreads();

  // logsoftmax
  float logsum_k = buffer[LOGSOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i] = input_k[i] - logsum_k;
}

__global__ void cunnx_SoftMaxTree_updateOutput_kernel(
  float *output, float *input, float* weight, float* bias, 
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
  float *output_k = output + k*nInput;
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
    float max_k = buffer[LOGSOFTMAX_THREADS];
    buffer[threadIdx.x] = 0;
    for (int i=i_start; i<i_end; i+=i_step)
      buffer[threadIdx.x] += __expf(input_k[i]-max_k);

    __syncthreads();

    // reduce
    if (threadIdx.x == 0)
    {
      float logsum_k = 0;
      for (int i=0; i<blockDim.x; i++)
        logsum_k += buffer[i];
      buffer[LOGSOFTMAX_THREADS] = max_k + __logf(logsum_k);
    }

    __syncthreads();

    // logsoftmax
    float logsum_k = buffer[LOGSOFTMAX_THREADS];
    for (int i=i_start; i<i_end; i+=i_step)
      output_k[i] = input_k[i] - logsum_k;
      
    /* Narrow */
    THTensor_(set)(nodeInter, nodeOutput);
    THTensor_(narrow)(nodeOutput, nodeInter, 0, childIdx, 1); 
    
    /* CAddTable (without log, would have been CMulTable) */
    narrowsum += THTensor_(get1d)(nodeOutput, 0);
    
    n += nChildren;
    /* Break when root is reached */
    if (parentId == rootId) 
    {
      break;
    }
    childId = parentId;
  }
  
  while (1) 
  {
    float z = input_k[i];
    float *nodeWeight = weight+
    if(buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

}

static int cunnx_SoftMaxTree_updateOutput(lua_State *L)
{ 
  THCudaTensor *input = luaT_checkudata(L, 2, "torch.CudaTensor");  
  THIntTensor *target = (THIntTensor*)luaT_checkudata(L, 3, "torch.IntTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId") - 1);
  
  THIntTensor *childParent = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.IntTensor");
  THIntTensor *parentChildren = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.IntTensor");
  
  THCudaTensor *metadata = luaT_getfieldcheckudata(L, 1, "_linearOutput", "torch.CudaTensor");
  THFloatTensor *buffer = luaT_getfieldcheckudata(L, 1, "_buffer", "torch.FloatTensor");
  
  THCudaTensor *
  THFloatTensor *
  
  THCudaTensor *logsoftOutput = luaT_getfieldcheckudata(L, 1, "_logSoftMaxOutput", "torch.CudaTensor");
  
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
  // we use these to keep intermediate results for later backprop
  THFloatTensor_resize2d(buffer, m, 5);
  THCudaTensor_resize2d(metadata, m, 5);
  THCudaTensor_resize1d(logsoftOutput, n);
  
  n, m = 0;
  /* Fill node metadata buffer */
  for(i = 0; i < input->size[0]; i++)
  {
    long childId = (long)(THIntTensor_get1d(target, i)) - 1;
    while(1)
    {
      long parentId, parentIdx, childIdx, nChildren;
      /* get next Node in Tree */
      THIntTensor_select(node, childParent, 0, childId);
      parentId = (long)(THIntTensor_get1d(node, 0)) - 1;
      childIdx = (long)(THIntTensor_get1d(node, 1)) - 1;
      
      THIntTensor_select(node, parentChildren, 0, parentId);
      parentIdx = (long)(THIntTensor_get1d(node, 0)) - 1;
      nChildren = (long)(THIntTensor_get1d(node, 1));
      
      THFloatTensor_set2d(buffer, (float)parentId, m, 0);
      THFloatTensor_set2d(buffer, (float)childIdx, m, 1);
      THFloatTensor_set2d(buffer, (float)parentIdx, m, 2);
      THFloatTensor_set2d(buffer, (float)nChildren, m, 3);
      THFloatTensor_set2d(buffer, (float)i, m, 4);
      
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
  
  THTensor_copy(metadata, buffer);
  
  /* matrix multiplies */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS); // each thread is an input index
  cunnx_SoftMaxTree_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(logsoftOutput), 
    THCudaTensor_data(input), THCudaTensor_data(weight), 
    THCudaTensor_data(bias), THCudaTensor_data(metadata), 
    input->size[0], input->size[1]
  );

  
  /* logsoftmax */
  
 
      
      // we use these to keep intermediate results for later backprop
      THTensor_(resize1d)(linearOutput, n+nChildren);
      THTensor_(resize1d)(logsoftOutput, n+nChildren);
  
      /* Linear */
      THTensor_(narrow)(nodeWeight, weight, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeBias, bias, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeOutput, linearOutput, 0, n, nChildren);
      
      THTensor_(addmv)(nodeOutput, 1, nodeBias, 1, nodeWeight, nodeInput);
      
      /* LogSoftMax */
      THTensor_(set)(nodeInter, nodeOutput);
      THTensor_(narrow)(nodeOutput, logsoftOutput, 0, n, nChildren);
      
      input_data = THTensor_(data)(nodeInter);
      output_data = THTensor_(data)(nodeOutput);
      
      accreal logsum = 0;
      real maxInput = -THInf;
      
      for(d = 0; d < nChildren; d++)
        maxInput = THMax(maxInput, input_data[d]);

      for(d = 0; d < nChildren; d++)
        logsum += THExpMinusApprox(maxInput-input_data[d]);
      logsum = maxInput + log(logsum);

      for(d = 0; d < nChildren; d++)
        output_data[d] = input_data[d] - logsum;
        
      /* Narrow */
      THTensor_(set)(nodeInter, nodeOutput);
      THTensor_(narrow)(nodeOutput, nodeInter, 0, childIdx, 1); 
      
      /* CAddTable (without log, would have been CMulTable) */
      narrowsum += THTensor_(get1d)(nodeOutput, 0);
      n += nChildren;
      /* Break when root is reached */
      if (parentId == rootId) 
      {
        break;
      }
      childId = parentId;
    }
    THTensor_(set1d)(output, i, narrowsum);  
  }
  
  THIntTensor_free(node);
  THTensor_(free)(nodeWeight);
  THTensor_(free)(nodeBias);
  THTensor_(free)(nodeOutput);
  THTensor_(free)(nodeInput);
  THTensor_(free)(nodeInter);
  
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
