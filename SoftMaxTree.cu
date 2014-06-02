#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAXTREE_THREADS 32
#define SOFTMAXTREE_MAXCHILDREN 100


__global__ void cunnx_SoftMaxTree_updateOutput_kernel(
  float *output, float* logsoftOutput, 
  float *input, float* weight, float* bias, 
  float* target, float* childParent, float* parentChildren, 
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
  float *node;
  int n = 0;
  
  // zero buffer
  buffer[tx] = 0;
  
  __syncthreads();

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
    
    if (tx == 0)
      output[k] = linearOutput[0];
      
    n += nChildren;
    /* Break when root is reached */
    if (parentId == rootId) 
    {
      break;
    }
    childId = parentId;
  }
  //if (tx == 0)
  //  output[k] = narrowsum;
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
    input->size[1], rootId
  );
  printf("here2 %f\n", THCudaTensor_get1d(logsoftOutput, 0));
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  printf("here3\n");
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
