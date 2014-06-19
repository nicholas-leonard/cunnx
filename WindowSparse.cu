#define WINDOWSPARSE_THREADS 128
#define WINDOWSPARSE_MAXBLOCKSIZE 512
#define WINDOWSPARSE_MINBLOCKSIZE 32
#define WINDOWSPARSE_STREAMS 8

  
static int cunnx_WindowSparse_updateOutput(lua_State *L)
{ 
  /* input, inputIndice, outputIndice, inputScale, outputScale, gradOutput*/
  // batchSize x inputWindowSize x inputSize
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  // batchSize x inputWindowSize
  THLongTensor *inputIndice = (THLongTensor*)luaT_checkudata(L, 3, "torch.LongTensor");
  THCudaTensor *inputScale = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  // batchSize x outputWindowSize
  THLongTensor *outputIndice = (THLongTensor*)luaT_checkudata(L, 4, "torch.LongTensor");
  THCudaTensor *outputScale = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  
  int batchedGemmMax = luaT_getfieldcheckint(L, 1, "batchedGemmMax");
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int batchSize, inputWindowSize, outputWindowSize;
  
  // nOutputBlock x nInputBlock x outputSize x inputSize
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  // nOutputBlock x outputSize
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  THCudaTensor* output_, *weight_, *_weight_, *input_;
  
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t streams[WINDOWSPARSE_STREAMS];
  
  float alpha = 1;
  float beta = 0;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] <= inputSize, 2, "invalid input size"); 
  luaL_argcheck(L, inputIndice->nDimension == 1, 3, "1D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 1, 4, "1D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  
  batchSize = input->size[0];
  inputWindowSize = input->size[1];
  outputWindowSize = outputScale->size[1];
  
  THCudaTensor_resize2d(output, input->size[0], outputScale->size[1]);
    
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) 
    THError("CUBLAS initialization failed");
    
  output_ = THCudaTensor_new();
  weight_ = THCudaTensor_new();
  _weight_ = THCudaTensor_new();
  input_ = THCudaTensor_new();
  
  if (sqrt(inputWindowSize*outputWindowSize) > batchedGemmMax)
  {
    for (int i=0; i<WINDOWSPARSE_STREAMS; i++)
    {
      if (cudaStreamCreate(&streams[i]) != cudaSuccess)
        THError("error initializing stream");
    }
  
    for (int i=0; i<batchSize; i++)
    {
      int inputIdx, outputIdx;
  
      cublasSetStream(handle, streams[i%WINDOWSPARSE_STREAMS]);
      
      inputIdx = THLongTensor_get1d(inputIndice, i);
      outputIdx = THLongTensor_get1d(outputIndice, i);
      
      THCudaTensor_select(output_, output, 0, i);
      THCudaTensor_select(input_, input, 0, i);
      THCudaTensor_narrow(_weight_, weight, 1, inputIdx, inputWindowSize);
      THCudaTensor_narrow(weight_, _weight_, 0, outputIdx, outputWindowSize);
      
      if(weight_->stride[0] == 1)
      {
        stat = cublasSgemv(handle, CUBLAS_OP_N, weight_->size[0], weight_->size[1],
                          &alpha, (const float*)THCudaTensor_data(weight_), weight_->stride[1],
                          (const float*)THCudaTensor_data(input_), input_->stride[0],
                          &beta, THCudaTensor_data(output_), output_->stride[0]);
      }
      else if(weight_->stride[1] == 1)
      {
        stat = cublasSgemv(handle, CUBLAS_OP_T,  weight_->size[1], weight_->size[0],
                          &alpha, (const float*)THCudaTensor_data(weight_), weight_->stride[0],
                          (const float*)THCudaTensor_data(input_), input_->stride[0],
                          &beta, THCudaTensor_data(output_), output_->stride[0]);
      }
      else
      {
        THError("expecting matrix with at least one contiguous dimension");
      }
    }

    cublasSetStream(handle, NULL);
    THCublasCheck();  
    
    for (int i=0; i<WINDOWSPARSE_STREAMS; i++)
    {
      if (cudaStreamDestroy(streams[i]) != cudaSuccess)
        THError("error destroying stream");
    }
  }
  else
  {  
    THCharTensor *inputHost = (THCharTensor*)luaT_getfieldcheckudata(L, 1, "inputHost", "torch.CharTensor");
    THCharTensor *weightHost = (THCharTensor*)luaT_getfieldcheckudata(L, 1, "weightHost", "torch.CharTensor");
    THCharTensor *outputHost = (THCharTensor*)luaT_getfieldcheckudata(L, 1, "outputHost", "torch.CharTensor");
    
    THCudaTensor *inputCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "inputCuda", "torch.CudaTensor");
    THCudaTensor *weightCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weightCuda", "torch.CudaTensor");
    THCudaTensor *outputCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "outputCuda", "torch.CudaTensor");
    
    THCharTensor_resize1d(inputHost, batchSize*sizeof(float*));
    THCharTensor_resize1d(weightHost, batchSize*sizeof(float*));
    THCharTensor_resize1d(outputHost, batchSize*sizeof(float*));
    
    THCudaTensor_resize1d(inputCuda, batchSize*sizeof(float*)/sizeof(float));
    THCudaTensor_resize1d(weightCuda, batchSize*sizeof(float*)/sizeof(float));
    THCudaTensor_resize1d(outputCuda, batchSize*sizeof(float*)/sizeof(float));
    
    const float **inputB = (const float **)THCharTensor_data(inputHost);
    const float **weightB = (const float **)THCharTensor_data(weightHost);
    float **outputB = (float **)THCharTensor_data(outputHost);
    
    const float **inputB_d = (const float **)THCudaTensor_data(inputCuda);
    const float **weightB_d = (const float **)THCudaTensor_data(weightCuda);
    float **outputB_d = (float **)THCudaTensor_data(outputCuda);
    
    for (int i=0; i<batchSize; i++)
    {
      int inputIdx, outputIdx;
      
      inputIdx = THLongTensor_get1d(inputIndice, i);
      outputIdx = THLongTensor_get1d(outputIndice, i);
      
      THCudaTensor_select(output_, output, 0, i);
      THCudaTensor_select(input_, input, 0, i);
      THCudaTensor_narrow(_weight_, weight, 1, inputIdx, inputWindowSize);
      THCudaTensor_narrow(weight_, _weight_, 0, outputIdx, outputWindowSize);
      
      inputB[i] = THCudaTensor_data(input_);
      weightB[i] = THCudaTensor_data(weight_);
      outputB[i] = THCudaTensor_data(output_);
    }
    
    if(cudaMemcpy(inputB_d, inputB, sizeof(float*) * batchSize, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    if(cudaMemcpy(weightB_d, weightB, sizeof(float*) * batchSize, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    if(cudaMemcpy(outputB_d, outputB, sizeof(float*) * batchSize, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    
                       
    stat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             outputWindowSize, 1, inputWindowSize,
                             &alpha, weightB_d, inputSize, 
                             inputB_d, inputWindowSize, 
                             &beta, outputB_d, outputWindowSize, 
                             batchSize);
    
    if (stat != CUBLAS_STATUS_SUCCESS) 
      THError("cublasSgemmBatched failed");
    
    
  }
  
  cublasDestroy(handle);
  THCublasCheck();  
  
  THCudaTensor_free(input_);
  THCudaTensor_free(weight_);
  THCudaTensor_free(_weight_);
  THCudaTensor_free(output_);
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
}


static int cunnx_WindowSparse_updateGradInput(lua_State *L)
{ 
  /* input, inputIndice, outputIndice, inputScale, outputScale*/
  // batchSize x inputWindowSize x inputSize
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  // batchSize x inputWindowSize
  THCudaTensor *inputIndice = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *inputScale = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  // batchSize x outputWindowSize
  THCudaTensor *outputIndice = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *outputScale = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 7, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  
  // nOutputBlock x nInputBlock x outputSize x inputSize
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_gradInput", "torch.CudaTensor");
  THCudaTensor *gradOutputScale = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradOutputScale", "torch.CudaTensor");
  
  luaL_argcheck(L, input->nDimension == 3, 2, "3D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[2] == inputSize, 2, "invalid input size"); 
  luaL_argcheck(L, inputIndice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndice = THCudaTensor_newContiguous(inputIndice);
  outputIndice = THCudaTensor_newContiguous(outputIndice); 
  inputScale = THCudaTensor_newContiguous(inputScale);
  outputScale = THCudaTensor_newContiguous(outputScale); 
  gradOutput = THCudaTensor_newContiguous(gradOutput);
  
  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_resizeAs(gradOutputScale, outputScale);
  
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  THCudaTensor_free(input);
  THCudaTensor_free(inputIndice);
  THCudaTensor_free(outputIndice);
  THCudaTensor_free(inputScale);
  THCudaTensor_free(outputScale);
  THCudaTensor_free(gradOutput);

  return 2;
}
  
__global__ void cunnx_WindowSparse_accGradParameters_kernel(
  float *gradWeight, float* gradBias, float *gradOutput, 
  float *input, float *inputIndice, float *outputIndice, 
  float *inputScale, float *outputScale,  
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize, float scale)
{
  __shared__ float buffer[WINDOWSPARSE_THREADS];
  __shared__ float gradOutputBuffer[WINDOWSPARSE_MAXBLOCKSIZE];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *input_k = input + k*inputWindowSize*inputSize;
  float *gradOutput_k = gradOutput + k*outputWindowSize*outputSize;
  float *inputIndice_k = inputIndice + k*inputWindowSize;
  float *outputIndice_k = outputIndice + k*outputWindowSize;
  float *inputScale_k = inputScale + k*inputWindowSize;
  float *outputScale_k = outputScale + k*outputWindowSize;
  
  // loop through blocks
  for (int m=0; m<outputWindowSize; m++)
  {
    int outputIdx = (int)outputIndice_k[m] - 1;
    float outputScale = outputScale_k[m];
    
    if (outputScale <= 0) // break on non-positive scale. 
      break;
      
    float *blockGradOutput = gradOutput_k + m*outputSize;
    float *blockGradBias = gradBias + outputIdx*outputSize;
    
    for (int j=tx; j<outputSize; j+=i_step)
    {
      gradOutputBuffer[j] = blockGradOutput[j]*outputScale*scale;
    }
    
    __syncthreads(); // needed for some reason
    
    for (int l=0; l<inputWindowSize; l++)
    {
      int inputIdx = (int)inputIndice_k[l] - 1;
      float inputScale = inputScale_k[l];
      
      if (inputScale <= 0) // break on non-positive scale. 
        break;
      
      float *blockInput = input_k + l*inputSize;
      float *blockGradWeight = gradWeight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;
      
      // addr weights (scalar-products)
      for (int i=tx; i<inputSize; i+=i_step)
      {
        // copy input to buffer
        buffer[tx] = blockInput[i]*inputScale;
      
        for (int j=0; j<outputSize; j++)
        {
          // multiply accumulate weights
          atomicAdd(&(blockGradWeight[j*inputSize + i]), gradOutputBuffer[j]*buffer[tx]);
        }
      }
    }
    
    __syncthreads(); // needed for some reason
    
    // cadd bias 
    for (int j=tx; j<outputSize; j+=i_step)
    {
       // multiply accumulate biases
      atomicAdd(&(blockGradBias[j]), gradOutputBuffer[j]);
    }    
  }
}


static int cunnx_WindowSparse_accGradParameters(lua_State *L)
{ 
  /* input, inputIndice, outputIndice, inputScale, outputScale, gradOutput, scale */
  // batchSize x inputWindowSize x inputSize
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  // batchSize x inputWindowSize
  THCudaTensor *inputIndice = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *inputScale = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  // batchSize x outputWindowSize
  THCudaTensor *outputIndice = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *outputScale = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 7, "torch.CudaTensor");
  float scale = luaL_optnumber(L, 8, 1);
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  int i, j, k;
  
  // nOutputBlock x nInputBlock x outputSize x inputSize
  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  
  THIntTensor *inputIndiceHost = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "inputIndiceHost", "torch.IntTensor");
  THIntTensor *outputIndiceHost = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceHost", "torch.IntTensor");
  THFloatTensor *inputScaleHost = (THFloatTensor*)luaT_getfieldcheckudata(L, 1, "inputScaleHost", "torch.FloatTensor");
  THFloatTensor *outputScaleHost = (THFloatTensor*)luaT_getfieldcheckudata(L, 1, "outputScaleHost", "torch.FloatTensor");
  
  luaL_argcheck(L, input->nDimension == 3, 2, "3D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[2] == inputSize, 2, "invalid input size"); 
  luaL_argcheck(L, inputIndice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndice = THCudaTensor_newContiguous(inputIndice);
  outputIndice = THCudaTensor_newContiguous(outputIndice); 
  inputScale = THCudaTensor_newContiguous(inputScale);
  outputScale = THCudaTensor_newContiguous(outputScale); 
  gradOutput = THCudaTensor_newContiguous(gradOutput);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(WINDOWSPARSE_THREADS);
  cunnx_WindowSparse_accGradParameters_kernel<<<blocks,threads>>>(
    THCudaTensor_data(gradWeight), THCudaTensor_data(gradBias), 
    THCudaTensor_data(gradOutput), THCudaTensor_data(input),
    THCudaTensor_data(inputIndice), THCudaTensor_data(outputIndice), 
    THCudaTensor_data(inputScale), THCudaTensor_data(outputScale), 
    inputSize, outputSize, nInputBlock, nOutputBlock, 
    inputIndice->size[1], outputIndice->size[1], scale
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
    
  // copy updated block indices from device to host
  THIntTensor_resize2d(inputIndiceHost, inputIndice->size[0], inputIndice->size[1]);
  THIntTensor_resize2d(outputIndiceHost, outputIndice->size[0], outputIndice->size[1]);
  THFloatTensor_resize2d(inputScaleHost, inputScale->size[0], inputScale->size[1]);
  THFloatTensor_resize2d(outputScaleHost, outputScale->size[0], outputScale->size[1]);
  
  THIntTensor_copyCuda(inputIndiceHost, inputIndice);
  THIntTensor_copyCuda(outputIndiceHost, outputIndice);
  THFloatTensor_copyCuda(inputScaleHost, inputScale);
  THFloatTensor_copyCuda(outputScaleHost, outputScale);
  
  lua_getfield(L, 1, "updates");
  
  // fill updates table
  for (k=0; k<input->size[0]; k++)
  {
    
    for (i=0; i<inputIndiceHost->size[1]; i++)
    {
      int inputIdx = THIntTensor_get2d(inputIndiceHost, k, i);
      float inputScale = THFloatTensor_get2d(inputScaleHost, k, i);
      
      if (inputScale <= 0)
        break;
     
      /* updates will contain inputIdx (key) sum of scales (value)*/
      lua_pushinteger(L, inputIdx);
      lua_gettable(L, -2);
      if lua_isnil(L, -1)
      {
        lua_pop(L, 1);
        lua_pushinteger(L, inputIdx); /* key */
        lua_newtable(L);  /* value */
        lua_settable(L, -3);
        
        lua_pushinteger(L, inputIdx);
        lua_gettable(L, -2);
      }
      
      for (j=0; j<outputIndiceHost->size[1]; j++)
      {
        int outputIdx = THIntTensor_get2d(outputIndiceHost, k, j);
        float outputScale = THFloatTensor_get2d(outputScaleHost, k, j);
        double count;
        
        if (outputScale <= 0)
          break;
        
        /* updates will contain inputIdx (key) sum of scales (value)*/
        lua_pushinteger(L, outputIdx);
        lua_gettable(L, -2);
        count = lua_tonumber(L, -1) + scale;
        lua_pop(L, 1);
        
        lua_pushinteger(L, outputIdx); /* key */
        lua_pushnumber(L, count); /* value */
        lua_settable(L, -3);
      }
      
      lua_pop(L, 1);
    }
  }
  
  THCudaTensor_free(input);
  THCudaTensor_free(inputIndice);
  THCudaTensor_free(outputIndice);
  THCudaTensor_free(inputScale);
  THCudaTensor_free(outputScale);
  THCudaTensor_free(gradOutput);

  return 0;
}  
  
static const struct luaL_Reg cunnx_WindowSparse__ [] = {
  {"WindowSparse_updateOutput", cunnx_WindowSparse_updateOutput},
  {"WindowSparse_updateGradInput", cunnx_WindowSparse_updateGradInput},
  {"WindowSparse_accGradParameters", cunnx_WindowSparse_accGradParameters},
  {NULL, NULL}
};

static void cunnx_WindowSparse_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_WindowSparse__, "nn");
  lua_pop(L,1);
}
