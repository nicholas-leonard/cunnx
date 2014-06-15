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
        CudaAssert(isfinite(buffer[tx]))
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
        CudaAssert(isfinite(buffer[0]))
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
      CudaAssert(isfinite(max_k))
      buffer[SOFTMAXTREE_THREADS] = max_k;
    }

    __syncthreads();
    
    // logadd?
    float max_k = buffer[SOFTMAXTREE_THREADS];
    buffer[tx] = 0;
    for (int i=tx; i<nChildren; i+=i_step)
    {
      buffer[tx] += expf(linearOutput[i]-max_k);
      CudaAssert(isfinite(buffer[tx]))
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
      CudaAssert(isfinite(m))
      buffer[SOFTMAXTREE_THREADS] = m;
    }

    __syncthreads();

    // logsoftmax
    float logsum_k = buffer[SOFTMAXTREE_THREADS];
    for (int i=tx; i<nChildren; i+=i_step)
    {
      nodeOutput[i] = linearOutput[i] - logsum_k;
      CudaAssert(isfinite(nodeOutput[i]))
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
    CudaAssert(isfinite(narrowsum))
  }
}


static int cunnx_SoftMaxTree_updateOutput(lua_State *L)
{ 
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
      CudaAssert(isfinite(nodeGrad[i]))
    }
    
    __syncthreads();
    if (tx == 0)
    {
      nodeGrad[childIdx] += grad;
      CudaAssert(isfinite(nodeGrad[childIdx]))
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
        CudaAssert(isfinite(buffer[tx]))
      }
      // accumulate into global memory
      gradInput_k[i] += buffer[tx];
      CudaAssert(isfinite(gradInput_k[i]))
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
  float *gradWeight, float *gradBias, float *input, 
  float *linearGradOutput, float *nodeUpdateCuda, float *target, 
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
  float *nodeUpdate = nodeUpdateCuda + maxDept*k; 
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
        CudaAssert(isfinite(dw))
        atomicAdd(&nodeGradWeight[j*nInput + i], dw);
      }
    }
    
    // cadd bias
    for (int j=tx; j<nChildren; j+=i_step)
    {
      // multiply accumulate biases
      float db = scale*nodeGradOutput[j];
      CudaAssert(isfinite(db))
      atomicAdd(&nodeGradBias[j], db);
    }
    
    // keep track of which node gets gradients
    nodeUpdate[m] = (float)parentId;
    
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
  THCudaTensor *nodeUpdateCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_nodeUpdateCuda", "torch.CudaTensor");
  THIntTensor *nodeUpdateHost = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "_nodeUpdateHost", "torch.IntTensor");
  
  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  
  int i, j;
  THIntTensor *nodeUpdate;
  
  lua_getfield(L, 1, "updates");
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  
  input = THCudaTensor_newContiguous(input); 
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each block is an example
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_accGradParameters_kernel<<<blocks,threads>>>(
    THCudaTensor_data(gradWeight), THCudaTensor_data(gradBias), 
    THCudaTensor_data(input), THCudaTensor_data(linearGradOutput), 
    THCudaTensor_data(nodeUpdateCuda), THCudaTensor_data(target), 
    THCudaTensor_data(childParent), THCudaTensor_data(parentChildren), 
    input->size[1], rootId, maxFamilyPath, maxDept, scale 
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  // copy updated nodeIds from device to host
  THIntTensor_copyCuda(nodeUpdateHost, nodeUpdateCuda);
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

__global__ void cunnx_SoftMaxTree_updateParameters_kernel(
  float *weight, float *bias, float *gradWeight, float *gradBias, 
  float *childParent, float *parentChildren, float *paramUpdateCuda,
  int nInput, float lr, float maxnorm)
{
  __shared__ float buffer[SOFTMAXTREE_THREADS];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int nodeId = paramUpdateCuda[blockIdx.x] - 1;
  int parentId, parentIdx, nChildren;
  float *nodeGradBias, *nodeBias;
  
  /* get next Node in Tree */
  float *node = childParent + nodeId*2;
  parentId = (int)node[0] - 1;
  
  node = parentChildren + parentId*2;
  parentIdx = (int)node[0] - 1;
  nChildren = (int)node[1];

  for (int j=0; j<nChildren; j++)
  {
    float *nodeWeight = weight + (parentIdx+j)*nInput;
    float *nodeGradWeight = gradWeight + (parentIdx+j)*nInput;
    
    buffer[tx] = 0;
    for (int i=tx; i<nInput; i+=i_step)
    {
      // update weights
      float w = nodeWeight[i];
      w -= nodeGradWeight[i]*lr;
      CudaAssert(isfinite(w))
      // norm of row
      buffer[tx] += w*w;
      nodeWeight[i] = w;
    }
    
    // add (reduce)
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
      __syncthreads();
      if (tx < stride)
        buffer[tx] += buffer[tx+stride];
    }
    
    // clip norms
    __syncthreads();
    float norm = sqrt(buffer[0]);
    if (norm > maxnorm) 
    {
      norm = maxnorm / (norm + 1e-7);
      CudaAssert(isfinite(norm))
      // renormalize
      for (int i=tx; i<nInput; i+=i_step)
      {
        nodeWeight[i] *= norm;
      }
    }
  }
    
  nodeGradBias = gradBias + parentIdx;
  nodeBias = bias + parentIdx;
  for (int j=tx; j<nChildren; j+=i_step)
  {
    // update biases
    nodeBias[j] -= nodeGradBias[j]*lr;
    CudaAssert(isfinite(nodeBias[j]))
  }
}

static int cunnx_SoftMaxTree_updateParameters(lua_State *L)
{
  float lr = (float)lua_tonumber(L, 2);   
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int rootId = luaT_getfieldcheckint(L, 1, "rootId") - 1;
  int maxFamilyPath = (int)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  int maxDept = luaT_getfieldcheckint(L, 1, "maxDept");
  float maxnorm = (float)luaT_getfieldcheckdouble(L, 1, "maxNorm");
  
  THCudaTensor *childParent = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "childParentCuda", "torch.CudaTensor");
  THCudaTensor *parentChildren = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "parentChildrenCuda", "torch.CudaTensor");
  
  THCudaTensor *linearGradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_multiBuffer", "torch.CudaTensor");
  THCudaTensor *paramUpdateCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_paramUpdateCuda", "torch.CudaTensor");
  THIntTensor *paramUpdateHost = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "_paramUpdateHost", "torch.IntTensor");
  
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
    
  int n = 0;
  
  /* table is in the stack at index -1 */
  lua_getfield(L, 1, "updates");
  lua_pushnil(L);  /* first key */
  while (lua_next(L, -2) != 0) {
    /* uses 'key' (at index -2) and 'value' (at index -1) */
    int nodeId = (int)lua_tonumber(L, -2);
    float scale = (float)lua_tonumber(L, -1);
    /* removes 'value'; keeps 'key' for next iteration */
    lua_pop(L, 1);
    // count number of elements in table
    n += 1;
  }
  
  if (n == 0) 
    return 0;
  
  THIntTensor_resize1d(paramUpdateHost, n);
  THCudaTensor_resize1d(paramUpdateCuda, n);
  
  /* table is in the stack at index -1 */
  lua_getfield(L, 1, "updates");
  lua_pushnil(L);  /* first key */
  n = 0;
  while (lua_next(L, -2) != 0) {
    /* uses 'key' (at index -2) and 'value' (at index -1) */
    int nodeId = (int)lua_tonumber(L, -2);
    float scale = (float)lua_tonumber(L, -1);
    /* removes 'value'; keeps 'key' for next iteration */
    lua_pop(L, 1);
    // add node to paramUpdate tensor
    THIntTensor_set1d(paramUpdateHost, n, nodeId);
    n += 1;
  }
  
  // send node indices to device
  THCudaTensor_copyInt(paramUpdateCuda, paramUpdateHost);
  
  /* call cudakernel */
  dim3 blocks(paramUpdateHost->size[0]); // each block is a node
  dim3 threads(SOFTMAXTREE_THREADS);
  cunnx_SoftMaxTree_updateParameters_kernel<<<blocks,threads>>>(
    THCudaTensor_data(weight), THCudaTensor_data(bias), 
    THCudaTensor_data(gradWeight), THCudaTensor_data(gradBias), 
    THCudaTensor_data(childParent), THCudaTensor_data(parentChildren), 
    THCudaTensor_data(paramUpdateCuda), weight->size[1], lr, maxnorm
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  return 0;
}

static const struct luaL_Reg cunnx_SoftMaxTree__ [] = {
  {"SoftMaxTree_updateOutput", cunnx_SoftMaxTree_updateOutput},
  {"SoftMaxTree_updateGradInput", cunnx_SoftMaxTree_updateGradInput},
  {"SoftMaxTree_accGradParameters", cunnx_SoftMaxTree_accGradParameters},
  {"SoftMaxTree_updateParameters", cunnx_SoftMaxTree_updateParameters},
  {NULL, NULL}
};

static void cunnx_SoftMaxTree_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_SoftMaxTree__, "nn");
  lua_pop(L,1);
}
