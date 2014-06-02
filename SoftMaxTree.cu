#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAXTREE_THREADS 32
#define SOFTMAXTREE_MAXCHILDREN 10000


__global__ void cunnx_SoftMaxTree_updateOutput_kernel(
  float *output, float* logsoftOutput,
  float *input, float* weight, float* bias, 
  float *target, float* childParent, float* parentChildren, 
  int nInput, int rootId, int maxFamilyPath)
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
  int childId = target[k] - 1;
  int parentId, parentIdx, childIdx, nChildren;
  int nOutput;
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
    
    /* Linear */
    
    nodeWeight = weight + parentIdx*nInput;
    nodeBias = bias + parentIdx;
    
    // addmv (dot products)
    for (int j=0; j<nChildren; j++)
    {
       // zero buffer
      buffer[tx] = 0;
      __syncthreads();
      
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
        linearOutput[j] = buffer[0] + nodeBias[j];
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
      
    __syncthreads();
    
    /* Narrow + CAddTable (without log, would have been CMulTable) */
    if (tx == 0)
      narrowsum += nodeOutput[childIdx];
      
    n += nChildren;
    /* Break when root is reached */
    if (parentId == rootId) 
    {
      break;
    }
    childId = parentId;
  }
  if (tx == 0)
    output[k] = narrowsum;
}


static int cunnx_SoftMaxTree_updateOutput(lua_State *L)
{ 
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamilyPath = (int)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.CudaTensor");

  THCudaTensor *logsoftOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");  
  
  input = THCudaTensor_newContiguous(input);
  THCudaTensor_resize1d(output, input->size[0]);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(logsoftOutput), 
    THCudaTensor_data(input), THCudaTensor_data(weight), 
    THCudaTensor_data(bias), THCudaTensor_data(target), 
    THCudaTensor_data(childParent), THCudaTensor_data(parentChildren), 
    input->size[1], rootId, maxFamilyPath
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  THCudaTensor_free(input);
  return 1;
}


__global__ void cunnx_SoftMaxTree_updateGradInput_kernel(
  float *gradInput, float* logsoftOutput, float *gradOutput, float* weight,
  float *target, float* childParent, float* parentChildren, 
  int nInput, int rootId, int maxFamilyPath)
{
  //__shared__ float input_buffer[nInput]; // constant might be faster
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

    for(int i = tx; i < nChildren; i+=i_step)
      nodeGrad[i] = -exp(nodeGrad[i])*grad;
    
    __syncthreads();
    if (tx == 0) // compare this to % childIdx
      nodeGrad[childIdx] += grad;

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
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamilyPath = (int)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.CudaTensor");
  
  THCudaTensor *logsoftOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");  
  
  luaL_argcheck(L, gradOutput->nDimension == 1, 2, "1D tensor expected");
  
  THCudaTensor_resizeAs(gradInput, input);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_updateGradInput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(gradInput), THCudaTensor_data(logsoftOutput), 
    THCudaTensor_data(gradOutput), THCudaTensor_data(weight), 
    THCudaTensor_data(target), THCudaTensor_data(childParent), 
    THCudaTensor_data(parentChildren), 
    input->size[1], rootId, maxFamilyPath
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  return 1;
}

__global__ void cunnx_SoftMaxTree_accGradParameters_kernel(
  float *gradWeight, float *gradBias, float *input, float* linearGradOutput,
  float *target, float* childParent, float* parentChildren, 
  float scale, int nInput, int rootId, int maxFamilyPath)
{
  __shared__ float buffer[SOFTMAXTREE_THREADS+1];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  float *input_k = input + k*nInput;
  float *nodeGradOutput, *nodeGradWeight, *nodeGradBias;
  int childId = target[k] - 1;
  int parentId, parentIdx, childIdx, nChildren;
  float *node;
  int n = 0;
  THIntTensor *node;
  
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
    
    nodeGradOutput = linearGradOutput + maxFamilyPath*k + n; 
    nodeGradWeight = gradWeight + parentIdx*nInput;
    nodeGradBias = gradBias + parentIdx;
      
    THTensor_(addr)(nodeGradWeight, 1, nodeGradWeight, scale, nodeGradOutput, nodeInput);
    THTensor_(cadd)(nodeGradBias, nodeGradBias, scale, nodeGradOutput);
    
    // addr weights (scalar-products)
    for (int i=tx; i<nInput; i+=i_step)
    {
      // copy input to buffer
      buffer[tx] = input_k[i];
    
      for (int j=0; j<nChildren; j++)
      {
        // multiply accumulate weights
        nodeGradWeight[j*nInput + i] += scale*nodeGrad[j]*buffer[tx];
      }
    }
    
    // cadd bias
    for (int i=tx; i<nChildren; i+=i_step)
    {
      // multiply accumulate weights
      nodeGradBias[i] += scale*nodeGrad[i]
    }
    
    n += nChildren;
    /* Break when root is reached */
    if (parentId == rootId)
    {
      break;
    }
    childId = parentId;
  }
}

static int cunnx_SoftMaxTree_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");  
  float scale = luaL_optnumber(L, 5, 1);
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamilyPath = (int)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.CudaTensor");
  
  THCudaTensor *linearGradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  
  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  
  lua_getfield(L, 1, "updates"); // this will be a pain to fill
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");  
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_accGradParameters_kernel<<<blocks,threads>>>(
    THCudaTensor_data(gradWeight), THCudaTensor_data(Bias), 
    THCudaTensor_data(input), THCudaTensor_data(linearGradOutput), 
    THCudaTensor_data(target), THCudaTensor_data(childParent), 
    THCudaTensor_data(parentChildren), 
    input->size[1], rootId, maxFamilyPath, scale
  );
    
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
