
static int cunnx_SoftMaxTree_updateOutput(lua_State *L)
{ 
  THCudaTensor *input = luaT_checkudata(L, 2, "torch.CudaTensor");  
  THIntTensor *target = (THIntTensor*)luaT_checkudata(L, 3, "torch.IntTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId") - 1);
  
  THIntTensor *childParent = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.IntTensor");
  THIntTensor *parentChildren = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.IntTensor");
  
  THCudaTensor *linearOutput = luaT_getfieldcheckudata(L, 1, "_linearOutput", "torch.CudaTensor");
  THCudaTensor *logsoftOutput = luaT_getfieldcheckudata(L, 1, "_logSoftMaxOutput", "torch.CudaTensor");
  
  THCudaTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  THIntTensor *node;
  THCudaTensor *nodeWeight, *nodeBias, *nodeOutput, *nodeInput, *nodeInter;
  real *input_data, *output_data;

  long i, d;
  long n = 0;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");

  node = THIntTensor_new();
  nodeWeight = THTensor_(new)();
  nodeBias = THCudaTensor_(new)();
  nodeOutput = THCudaTensor_(new)();
  nodeInput = THCudaTensor_(new)();
  nodeInter = THCudaTensor_(new)();
  
  
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
