#define BLOCKSPARSE_THREADS 32
#define BLOCKSPARSE_MAXOUTPUTBLOCKSIZE 512
#define BLOCKSPARSE_STREAMS 8
  
__global__ void cunnx_BlockSparse_updateOutput_kernel(
  float *output, const float *input, const float *outputIndice, 
  const float *outputScale, const float *bias,  
  int outputSize, int nOutputBlock, 
  int inputWindowSize, int outputWindowSize)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *output_k = output + k*outputWindowSize*outputSize;
  const float *input_k = input + k*inputWindowSize*outputWindowSize*outputSize;
  const float *outputIndice_k = outputIndice + k*outputWindowSize;
  const float *outputScale_k = outputScale + k*outputWindowSize;
  
  for (int m=0; m<outputWindowSize; m++)
  {
    int outputIdx = (int)outputIndice_k[m] - 1;
    float outputScale = outputScale_k[m];
    
    for (int j=tx; j<outputSize; j+=i_step)
    {
      buffer[tx] = bias[outputIdx*outputSize + j];
          
      for (int l=0; l<inputWindowSize; l++)
        buffer[tx] += input_k[l*outputWindowSize*outputSize + m*outputSize + j];
      
      output_k[m*outputSize + j] = outputScale*buffer[tx];
    }
  }
}

static int cunnx_BlockSparse_updateOutput(lua_State *L)
{ 
  /* input, inputIndice, outputIndice, inputScale, outputScale, gradOutput*/
  // batchSize x inputWindowSize x inputSize
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
  // batchSize x inputWindowSize
  THCudaTensor *inputIndice = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *inputScale = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  // batchSize x outputWindowSize
  THCudaTensor *outputIndice = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *outputScale = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int inputWindowSize = luaT_getfieldcheckint(L, 1, "inputWindowSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  int batchedGemmMax = luaT_getfieldcheckint(L, 1, "batchedGemmMax");
  long nBatched = batchSize*inputWindowSize*outputWindowSize;
  
  THLongTensor *inputIndiceHost = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "inputIndiceHost", "torch.LongTensor");
  THLongTensor *outputIndiceHost = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceHost", "torch.LongTensor");
  // nOutputBlock x nInputBlock x outputSize x inputSize
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  // nOutputBlock x outputSize
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  // batchSize x inputWindowSize x outputWindowSize x outputSize
  THCudaTensor *outputBatched = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "outputBatched", "torch.CudaTensor");
  // batchSize x outputWindowSize x outputSize
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_output", "torch.CudaTensor");
  
  cublasStatus_t stat;
  cublasHandle_t handle;
  
  float alpha = 1;
  float beta = 0;
  
  if (nInputBlock > 1) 
  {
    luaL_argcheck(L, input->nDimension == 3, 2, "3D(batch mode) tensor expected");
    luaL_argcheck(L, input->size[2] == inputSize, 2, "invalid input size"); 
  } 
  else 
  {
    luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
    luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  }
  luaL_argcheck(L, inputIndice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "Expecting contiguous input");
  
  THCudaTensor_resize4d(outputBatched, batchSize, inputWindowSize, outputWindowSize, outputSize);
  THLongTensor_resize2d(inputIndiceHost, batchSize, inputWindowSize);
  THLongTensor_resize2d(outputIndiceHost, batchSize, outputWindowSize);
  
  THLongTensor_copyCuda(inputIndiceHost, inputIndice);
  THLongTensor_copyCuda(outputIndiceHost, outputIndice);
  
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) 
    THError("CUBLAS initialization failed");
  
  if ( nOutputBlock > 1 )
    THCudaTensor_resize3d(output, batchSize, outputWindowSize, outputSize);
  else
    THCudaTensor_resize2d(output, batchSize, outputSize);
  
  /* streamed or batched */
  if (sqrt(inputSize*outputSize) > batchedGemmMax)
  {
    cudaStream_t streams[BLOCKSPARSE_STREAMS];
    
    for (int i=0; i<BLOCKSPARSE_STREAMS; i++)
    {
      if (cudaStreamCreate(&streams[i]) != cudaSuccess)
        THError("error initializing stream");
    }
    cudaDeviceSynchronize();
    
    long batchedIdx = 0;
    for (int i=0; i<batchSize; i++)
    {
      float *inputPtr = THCudaTensor_data(input)+i*input->stride[0];
      float *outputPtr = THCudaTensor_data(outputBatched)+i*outputBatched->stride[0];
      long *inputIdxPtr = THLongTensor_data(inputIndiceHost)+i*inputIndiceHost->stride[0];
      long *outputIdxPtr = THLongTensor_data(outputIndiceHost)+i*outputIndiceHost->stride[0];
      
      for (int l=0; l<inputWindowSize; l++) 
      {              
        for (int m=0; m<outputWindowSize; m++)
        {
          cublasSetStream(handle, streams[batchedIdx%BLOCKSPARSE_STREAMS]);
      
          stat = cublasSgemv(handle, CUBLAS_OP_T,  inputSize, outputSize,
                            &alpha, (const float*)THCudaTensor_data(weight)+(inputIdxPtr[l]-1)*weight->stride[1] + (outputIdxPtr[m]-1)*weight->stride[0], inputSize,
                            (const float*)inputPtr, 1,
                            &beta, outputPtr, 1);
                            
          if (stat != CUBLAS_STATUS_SUCCESS) 
            THError("cublasSgemv failed");

          outputPtr += outputBatched->stride[2];
          batchedIdx++;
        }
        
        inputPtr += input->stride[1];
      }
    }
    
    cublasSetStream(handle, NULL);
    cudaDeviceSynchronize();
    
    for (int i=0; i<BLOCKSPARSE_STREAMS; i++)
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
  
    // put output back on top of the stack
    output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_output", "torch.CudaTensor");
    
    cublasSetStream(handle, NULL);
    
    THCharTensor_resize1d(inputHost, nBatched*sizeof(float*));
    THCharTensor_resize1d(weightHost, nBatched*sizeof(float*));
    THCharTensor_resize1d(outputHost, nBatched*sizeof(float*));
    
    THCudaTensor_resize1d(inputCuda, nBatched*sizeof(float*)/sizeof(float));
    THCudaTensor_resize1d(weightCuda, nBatched*sizeof(float*)/sizeof(float));
    THCudaTensor_resize1d(outputCuda, nBatched*sizeof(float*)/sizeof(float));
    
    const float **inputB = (const float **)THCharTensor_data(inputHost);
    const float **weightB = (const float **)THCharTensor_data(weightHost);
    float **outputB = (float **)THCharTensor_data(outputHost);
    
    const float **inputB_d = (const float **)THCudaTensor_data(inputCuda);
    const float **weightB_d = (const float **)THCudaTensor_data(weightCuda);
    float **outputB_d = (float **)THCudaTensor_data(outputCuda);
    
    long batchedIdx = 0;
    for (int i=0; i<batchSize; i++)
    {
      float *inputPtr = THCudaTensor_data(input)+i*input->stride[0];
      float *outputPtr = THCudaTensor_data(outputBatched)+i*outputBatched->stride[0];
      long *inputIdxPtr = THLongTensor_data(inputIndiceHost)+i*inputIndiceHost->stride[0];
      long *outputIdxPtr = THLongTensor_data(outputIndiceHost)+i*outputIndiceHost->stride[0];
      
      for (int l=0; l<inputWindowSize; l++) 
      {              
        for (int m=0; m<outputWindowSize; m++)
        {
          inputB[batchedIdx] = inputPtr;
          weightB[batchedIdx] = THCudaTensor_data(weight) + (outputIdxPtr[m]-1)*weight->stride[0] + (inputIdxPtr[l]-1)*weight->stride[1];
          outputB[batchedIdx] = outputPtr;

          outputPtr += outputBatched->stride[2];
          batchedIdx++;
        }
        
        inputPtr += input->stride[1];
      }
    }
    
    if(cudaMemcpy(inputB_d, inputB, sizeof(float*) * nBatched, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    if(cudaMemcpy(weightB_d, weightB, sizeof(float*) * nBatched, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    if(cudaMemcpy(outputB_d, outputB, sizeof(float*) * nBatched, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    
    stat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             outputSize, 1, inputSize,
                             &alpha, weightB_d, inputSize, 
                             inputB_d, inputSize, 
                             &beta, outputB_d, outputSize, 
                             nBatched);
    
    if (stat != CUBLAS_STATUS_SUCCESS) 
      THError("cublasSgemmBatched failed");
    
  }
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_updateOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(output), THCudaTensor_data(outputBatched), 
    THCudaTensor_data(outputIndice), THCudaTensor_data(outputScale),
    THCudaTensor_data(bias),  outputSize, nOutputBlock,
    inputWindowSize, outputWindowSize
  );
  
  cublasDestroy(handle);
  THCublasCheck();  
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
}
  
__global__ void cunnx_BlockSparse_updateGradOutput_kernel(
  float *_gradOutput, float* gradOutputScale, const float *gradOutput, 
  const float *output, const float *outputScale, 
  int outputWindowSize, int outputSize)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *_gradOutput_k = _gradOutput + k*outputWindowSize*outputSize;
  float *gradOutputScale_k = gradOutputScale + k*outputWindowSize;
  const float *gradOutput_k = gradOutput + k*outputWindowSize*outputSize;
  const float *output_k = output + k*outputWindowSize*outputSize;
  const float *outputScale_k = outputScale + k*outputWindowSize;
  
  
  // get gradients for outputScale (to be backwarded to a Gater)
  for (int m=0; m<outputWindowSize; m++)
  {
    float outputScale = outputScale_k[m];
    
    float *_blockGradOutput = _gradOutput_k + m*outputSize;  
    const float *blockGradOutput = gradOutput_k + m*outputSize;
    const float *blockOutput = output_k + m*outputSize;
    
    buffer[tx] = 0;
    
    for (int j=tx; j<outputSize; j+=i_step)
    {
      const float grad = blockGradOutput[j];
      buffer[tx] += blockOutput[j]*grad;
      _blockGradOutput[j] = grad*outputScale;
    }
    
    // add (reduce)
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
      __syncthreads();
      if (tx < stride)
        buffer[tx] += buffer[tx+stride];
    }
    
    if (tx == 0)
      gradOutputScale_k[m] = buffer[0]/outputScale;
  }
}


static int cunnx_BlockSparse_updateGradInput(lua_State *L)
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
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_output", "torch.CudaTensor");
  
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int inputWindowSize = luaT_getfieldcheckint(L, 1, "inputWindowSize");
  int outputWindowSize = luaT_getfieldcheckint(L, 1, "outputWindowSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  int batchedGemmMax = luaT_getfieldcheckint(L, 1, "batchedGemmMax");
  long nBatched = batchSize*inputWindowSize*outputWindowSize;
  
  THLongTensor *inputIndiceHost = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "inputIndiceHost", "torch.LongTensor");
  THLongTensor *outputIndiceHost = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceHost", "torch.LongTensor");
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInputBatched = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInputBatched", "torch.CudaTensor");
  THCudaTensor *_gradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_gradOutput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_gradInput", "torch.CudaTensor");
  THCudaTensor *gradOutputScale = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradOutputScale", "torch.CudaTensor");
  
  cublasStatus_t stat;
  cublasHandle_t handle;
  
  float alpha = 1;
  float beta = 0;
  
  if (nInputBlock > 1) 
  {
    luaL_argcheck(L, input->nDimension == 3, 2, "3D(batch mode) tensor expected");
    luaL_argcheck(L, input->size[2] == inputSize, 2, "invalid input size"); 
  } 
  else 
  {
    luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
    luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  }
  luaL_argcheck(L, inputIndice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "Expecting contiguous input");
  
  THCudaTensor_resizeAs(_gradOutput, gradOutput);
  THCudaTensor_resizeAs(gradOutputScale, outputScale);
  THCudaTensor_resize4d(gradInputBatched, batchSize, outputWindowSize, inputWindowSize, inputSize);
 
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_updateGradOutput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(_gradOutput), THCudaTensor_data(gradOutputScale), 
    THCudaTensor_data(gradOutput), THCudaTensor_data(output),
    THCudaTensor_data(outputScale), outputWindowSize, outputSize
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
    
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) 
    THError("CUBLAS initialization failed");
  
  /* streamed or batched */
  if (sqrt(inputSize*outputSize) > batchedGemmMax)
  {
    cudaStream_t streams[BLOCKSPARSE_STREAMS];
    
    for (int i=0; i<BLOCKSPARSE_STREAMS; i++)
    {
      if (cudaStreamCreate(&streams[i]) != cudaSuccess)
        THError("error initializing stream");
    }
    cudaDeviceSynchronize();
    
    long batchedIdx = 0;
    for (int i=0; i<batchSize; i++)
    {
      float *gradOutputPtr = THCudaTensor_data(_gradOutput)+i*_gradOutput->stride[0];
      float *gradInputPtr = THCudaTensor_data(gradInputBatched)+i*gradInputBatched->stride[0];
      long *inputIdxPtr = THLongTensor_data(inputIndiceHost)+i*inputIndiceHost->stride[0];
      long *outputIdxPtr = THLongTensor_data(outputIndiceHost)+i*outputIndiceHost->stride[0];
      
      for (int m=0; m<outputWindowSize; m++)
      {              
        for (int l=0; l<inputWindowSize; l++) 
        {
          cublasSetStream(handle, streams[batchedIdx%BLOCKSPARSE_STREAMS]);
      
          stat = cublasSgemv(handle, CUBLAS_OP_N,  inputSize, outputSize,
                            &alpha, (const float*)THCudaTensor_data(weight)+(outputIdxPtr[m]-1)*weight->stride[0]+(inputIdxPtr[l]-1)*weight->stride[1], inputSize,
                            (const float*)gradOutputPtr, 1,
                            &beta, gradInputPtr, 1);
                            
          if (stat != CUBLAS_STATUS_SUCCESS) 
            THError("cublasSgemv failed");

          gradInputPtr += gradInputBatched->stride[2];
          batchedIdx++;
        }
        
        gradOutputPtr += _gradOutput->stride[1];
      }
    }
    
    cublasSetStream(handle, NULL);
    cudaDeviceSynchronize();
    
    for (int i=0; i<BLOCKSPARSE_STREAMS; i++)
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
    // put gradInput back on top of the stack
    gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_gradInput", "torch.CudaTensor");
    gradOutputScale = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradOutputScale", "torch.CudaTensor");
    
    cublasSetStream(handle, NULL);
    
    THCharTensor_resize1d(inputHost, nBatched*sizeof(float*));
    THCharTensor_resize1d(weightHost, nBatched*sizeof(float*));
    THCharTensor_resize1d(outputHost, nBatched*sizeof(float*));
    
    THCudaTensor_resize1d(inputCuda, nBatched*sizeof(float*)/sizeof(float));
    THCudaTensor_resize1d(weightCuda, nBatched*sizeof(float*)/sizeof(float));
    THCudaTensor_resize1d(outputCuda, nBatched*sizeof(float*)/sizeof(float));
    
    float **gradInputB = (float **)THCharTensor_data(inputHost);
    const float **weightB = (const float **)THCharTensor_data(weightHost);
    const float **gradOutputB = (const float **)THCharTensor_data(outputHost);
    
    float **gradInputB_d = (float **)THCudaTensor_data(inputCuda);
    const float **weightB_d = (const float **)THCudaTensor_data(weightCuda);
    const float **gradOutputB_d = (const float **)THCudaTensor_data(outputCuda);
    

    long batchedIdx = 0;
    for (int i=0; i<batchSize; i++)
    {
      float *gradOutputPtr = THCudaTensor_data(_gradOutput)+i*_gradOutput->stride[0];
      float *gradInputPtr = THCudaTensor_data(gradInputBatched)+i*gradInputBatched->stride[0];
      long *inputIdxPtr = THLongTensor_data(inputIndiceHost)+i*inputIndiceHost->stride[0];
      long *outputIdxPtr = THLongTensor_data(outputIndiceHost)+i*outputIndiceHost->stride[0];
      
      for (int m=0; m<outputWindowSize; m++)
      {              
        for (int l=0; l<inputWindowSize; l++) 
        {
          gradInputB[batchedIdx] = gradInputPtr;
          weightB[batchedIdx] = THCudaTensor_data(weight)+(outputIdxPtr[m]-1)*weight->stride[0]+(inputIdxPtr[l]-1)*weight->stride[1];
          gradOutputB[batchedIdx] = gradOutputPtr;

          gradInputPtr += gradInputBatched->stride[2];
          batchedIdx++;
        }
        
        gradOutputPtr += _gradOutput->stride[1];
      }
    }
    
    if(cudaMemcpy(gradInputB_d, gradInputB, sizeof(float*)*nBatched, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    if(cudaMemcpy(weightB_d, weightB, sizeof(float*)*nBatched, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");
    if(cudaMemcpy(gradOutputB_d, gradOutputB, sizeof(float*)*nBatched, cudaMemcpyHostToDevice) != cudaSuccess)
      THError("cudaMemcpy failed");

    stat = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             inputSize, 1, outputSize,
                             &alpha, weightB_d, inputSize, 
                             gradOutputB_d, outputSize, 
                             &beta, gradInputB_d, inputSize, 
                             nBatched);
    
    if (stat != CUBLAS_STATUS_SUCCESS) 
      THError("cublasSgemmBatched failed");
    
  }
  
  THCudaTensor_sum(gradInput, gradInputBatched, 1);
  THCudaTensor_resizeAs(gradInput, input); 
  
  errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 2;
}
  
__global__ void cunnx_BlockSparse_accGradParameters_kernel(
  float *gradWeight, float* gradBias, float *gradOutput, 
  float *input, float *inputIndice, float *outputIndice, 
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize, float scale)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  __shared__ float gradOutputBuffer[BLOCKSPARSE_MAXOUTPUTBLOCKSIZE];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *input_k = input + k*inputWindowSize*inputSize;
  float *gradOutput_k = gradOutput + k*outputWindowSize*outputSize;
  float *inputIndice_k = inputIndice + k*inputWindowSize;
  float *outputIndice_k = outputIndice + k*outputWindowSize;
  
  // loop through blocks
  for (int m=0; m<outputWindowSize; m++)
  {
    int outputIdx = (int)outputIndice_k[m] - 1;
      
    float *blockGradOutput = gradOutput_k + m*outputSize;
    float *blockGradBias = gradBias + outputIdx*outputSize;
    
    for (int j=tx; j<outputSize; j+=i_step)
      gradOutputBuffer[j] = blockGradOutput[j]*scale;
    
    __syncthreads(); // needed for some reason
    
    for (int l=0; l<inputWindowSize; l++)
    {
      int inputIdx = (int)inputIndice_k[l] - 1;
      
      float *blockInput = input_k + l*inputSize;
      float *blockGradWeight = gradWeight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;
      
      // addr weights (scalar-products)
      for (int i=tx; i<inputSize; i+=i_step)
      {
        // copy input to buffer
        buffer[tx] = blockInput[i];
      
        // multiply accumulate weights
        for (int j=0; j<outputSize; j++)
          atomicAdd(&(blockGradWeight[j*inputSize + i]), gradOutputBuffer[j]*buffer[tx]);
      }
    }
    
    __syncthreads(); // needed for some reason
    
    // multiply accumulate biases 
    for (int j=tx; j<outputSize; j+=i_step)
      atomicAdd(&(blockGradBias[j]), gradOutputBuffer[j]);
  }
}


static int cunnx_BlockSparse_accGradParameters(lua_State *L)
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
  float scale = luaL_optnumber(L, 8, 1);
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  
  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  THCudaTensor *_gradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_gradOutput", "torch.CudaTensor");
  THLongTensor *inputIndiceHost = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "inputIndiceHost", "torch.LongTensor");
  THLongTensor *outputIndiceHost = (THLongTensor*)luaT_getfieldcheckudata(L, 1, "outputIndiceHost", "torch.LongTensor");
  
  if (nInputBlock > 1) 
  {
    luaL_argcheck(L, input->nDimension == 3, 2, "3D(batch mode) tensor expected");
    luaL_argcheck(L, input->size[2] == inputSize, 2, "invalid input size"); 
  } 
  else 
  {
    luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
    luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size"); 
  }
  luaL_argcheck(L, inputIndice->nDimension == 2, 3, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputIndice->nDimension == 2, 4, "2D(batch mode) tensor expected");
  luaL_argcheck(L, inputScale->nDimension == 2, 5, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputScale->nDimension == 2, 6, "2D(batch mode) tensor expected");
  luaL_argcheck(L, outputSize <= BLOCKSPARSE_MAXOUTPUTBLOCKSIZE, 1, "outputSize is too large");
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_accGradParameters_kernel<<<blocks,threads>>>(
    THCudaTensor_data(gradWeight), THCudaTensor_data(gradBias), 
    THCudaTensor_data(_gradOutput), THCudaTensor_data(input),
    THCudaTensor_data(inputIndice), THCudaTensor_data(outputIndice), 
    inputSize, outputSize, nInputBlock, nOutputBlock, 
    inputIndice->size[1], outputIndice->size[1], scale
  );
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  
  return 0;
}


  
static const struct luaL_Reg cunnx_BlockSparse__ [] = {
  {"BlockSparse_updateOutput", cunnx_BlockSparse_updateOutput},
  {"BlockSparse_updateGradInput", cunnx_BlockSparse_updateGradInput},
  {"BlockSparse_accGradParameters", cunnx_BlockSparse_accGradParameters},
  {NULL, NULL}
};

static void cunnx_BlockSparse_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_BlockSparse__, "nn");
  lua_pop(L,1);
}
