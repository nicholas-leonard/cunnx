#define BLOCKSPARSE_THREADS 32
#define BLOCKSPARSE_MAXBLOCKSIZE 1024
#define BLOCKSPARSE_MINBLOCKSIZE 32
#define BLOCKSPARSE_MINBLOCKS 32
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
  
  THCudaTensor *__output__, *_output_, *output_, *weight_, *_weight_, *_input_, *input_;
  
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
  
  __output__ = THCudaTensor_new();
  _output_ = THCudaTensor_new();
  output_ = THCudaTensor_new();
  weight_ = THCudaTensor_new();
  _weight_ = THCudaTensor_new();
  _input_ = THCudaTensor_new();
  input_ = THCudaTensor_new();
  
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
    
    for (int i=0; i<batchSize; i++)
    {
      THCudaTensor_select(_input_, input, 0, i);
      THCudaTensor_select(__output__, outputBatched, 0, i);
      
      for (int l=0; l<inputWindowSize; l++) 
      {
        int inputIdx = THLongTensor_get2d(inputIndiceHost, i, l) - 1;
        THCudaTensor_select(input_, _input_, 0, l);
        THCudaTensor_select(_output_, __output__, 0, l);
        
        for (int m=0; m<outputWindowSize; m++)
        {
          int outputIdx = THLongTensor_get2d(outputIndiceHost, i, m) - 1;
          THCudaTensor_select(output_, _output_, 0, m);
          
          THCudaTensor_select(_weight_, weight, 1, inputIdx);
          THCudaTensor_select(weight_, _weight_, 0, outputIdx);
          
          cublasSetStream(handle, streams[i%BLOCKSPARSE_STREAMS]);
      
          stat = cublasSgemv(handle, CUBLAS_OP_T,  inputSize, outputSize,
                            &alpha, (const float*)THCudaTensor_data(weight_), inputSize,
                            (const float*)THCudaTensor_data(input_), 1,
                            &beta, THCudaTensor_data(output_), 1);
                            
          if (stat != CUBLAS_STATUS_SUCCESS) 
            THError("cublasSgemv failed");
                            
        }
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
          long batchedIdx = i*inputWindowSize*outputWindowSize + l*outputWindowSize + m;

          inputB[batchedIdx] = inputPtr;
          weightB[batchedIdx] = THCudaTensor_data(weight)+(inputIdxPtr[l]-1)*weight->stride[1] + (outputIdxPtr[m]-1)*weight->stride[0];
          outputB[batchedIdx] = outputPtr;

          outputPtr += output->stride[2];
        }
        
        inputPtr += input->stride[1];
        outputPtr += output->stride[1];
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
  
  THCudaTensor_free(_input_);
  THCudaTensor_free(input_);
  THCudaTensor_free(weight_);
  THCudaTensor_free(_weight_);
  THCudaTensor_free(__output__);
  THCudaTensor_free(_output_);
  THCudaTensor_free(output_);
  
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
}


  
__global__ void cunnx_BlockSparse_updateGradInput_kernel(
  float *gradInput, float* gradOutputScale, float *gradOutput, 
  float *output, float *inputIndice, float *outputIndice, 
  float *inputScale, float *outputScale, float *weight,  
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  __shared__ float gradInputBuffer[BLOCKSPARSE_MAXBLOCKSIZE];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  int k = blockIdx.x;
  
  float *gradInput_k = gradInput + k*inputWindowSize*inputSize;
  float *gradOutput_k = gradOutput + k*outputWindowSize*outputSize;
  float *output_k = output + k*outputWindowSize*outputSize;
  float *inputIndice_k = inputIndice + k*inputWindowSize;
  float *outputIndice_k = outputIndice + k*outputWindowSize;
  float *inputScale_k = inputScale + k*inputWindowSize;
  float *outputScale_k = outputScale + k*outputWindowSize;
  float *gradOutputScale_k = gradOutputScale + k*outputWindowSize;
  
  // 1. loop through blocks to get gradInput
  for (int l=0; l<inputWindowSize; l++)
  {
    int inputIdx = (int)inputIndice_k[l] - 1;
    float inputScale = inputScale_k[l];
    
    if (inputScale <= 0) // break on non-positive scale. 
      break;
    
    float *blockGradInput = gradInput_k + l*inputSize;
      
    for (int j=tx; j<inputSize; j+=i_step)
    {
      gradInputBuffer[j] = 0;
    }
    
    for (int m=0; m<outputWindowSize; m++)
    {
      int outputIdx = (int)outputIndice_k[m] - 1;
      float outputScale = outputScale_k[m];
      
      if (outputScale <= 0) // break on non-positive scale. 
        break;
        
      float *blockGradOutput = gradOutput_k + m*outputSize;     
      float *blockWeight = weight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;
      
      // addmv (dot products)
      for (int i=tx; i<inputSize; i+=i_step)
      {
       // zero buffer
        buffer[tx] = 0;
        
        for (int j=0; j<outputSize; j++)
        {
          // multiply
          buffer[tx] += blockGradOutput[j]*blockWeight[j*inputSize + i]*outputScale;
        }
        
        // accumulate 
        gradInputBuffer[i] += buffer[tx];
      }
      
    }
    
    __syncthreads();
      
    for (int i=tx; i<inputSize; i+=i_step)
    {
      blockGradInput[i] = gradInputBuffer[i];
    }
    
    __syncthreads();
    
  }
  
  // 2. get gradients for outputScale (to be backwarded to a Gater)
  for (int m=0; m<outputWindowSize; m++)
  {
    float outputScale = outputScale_k[m];
    if (outputScale <= 0) // break on non-positive scale. 
      break;
      
    float *blockGradOutput = gradOutput_k + m*outputSize;
    float *blockOutput = output_k + m*outputSize;
    
    buffer[tx] = 0;
    
    for (int j=tx; j<outputSize; j+=i_step)
    {
      buffer[tx] += blockOutput[j]*blockGradOutput[j];
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
      gradOutputScale_k[m] = buffer[0];
    }
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
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndice = THCudaTensor_newContiguous(inputIndice);
  outputIndice = THCudaTensor_newContiguous(outputIndice); 
  inputScale = THCudaTensor_newContiguous(inputScale);
  outputScale = THCudaTensor_newContiguous(outputScale); 
  gradOutput = THCudaTensor_newContiguous(gradOutput);
  
  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_resizeAs(gradOutputScale, outputScale);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_updateGradInput_kernel<<<blocks,threads>>>(
    THCudaTensor_data(gradInput), THCudaTensor_data(gradOutputScale), 
    THCudaTensor_data(gradOutput), THCudaTensor_data(output),
    THCudaTensor_data(inputIndice), THCudaTensor_data(outputIndice), 
    THCudaTensor_data(inputScale), THCudaTensor_data(outputScale), 
    THCudaTensor_data(weight), inputSize, outputSize, nInputBlock, 
    nOutputBlock, inputIndice->size[1], outputIndice->size[1]
  );
  
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
  
__global__ void cunnx_BlockSparse_accGradParameters_kernel(
  float *gradWeight, float* gradBias, float *gradOutput, 
  float *input, float *inputIndice, float *outputIndice, 
  float *inputScale, float *outputScale,  
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock,
  int inputWindowSize, int outputWindowSize, float scale)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  __shared__ float gradOutputBuffer[BLOCKSPARSE_MAXBLOCKSIZE];
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
  
  // expect contiguous inputs
  input = THCudaTensor_newContiguous(input);
  inputIndice = THCudaTensor_newContiguous(inputIndice);
  outputIndice = THCudaTensor_newContiguous(outputIndice); 
  inputScale = THCudaTensor_newContiguous(inputScale);
  outputScale = THCudaTensor_newContiguous(outputScale); 
  gradOutput = THCudaTensor_newContiguous(gradOutput);
  
  /* call cudakernel */
  dim3 blocks(input->size[0]); // each cuda-block is an example
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_accGradParameters_kernel<<<blocks,threads>>>(
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



__global__ void cunnx_BlockSparse_updateParameters_kernel(
  float *weight, float *bias, float *gradWeight, float *gradBias, 
  float *paramUpdateCuda, float lr, float maxnorm, int nUpdate,
  int inputSize, int outputSize, int nInputBlock, int nOutputBlock)
{
  __shared__ float buffer[BLOCKSPARSE_THREADS];
  int tx = threadIdx.x;
  int i_step = blockDim.x;
  
  // update blockWeight and renorm
  for (int k=blockIdx.x; k<nUpdate; k+=gridDim.x)
  {
    int inputIdx = (int)paramUpdateCuda[k];
    int outputIdx = (int)paramUpdateCuda[gridDim.x + k];
    
    float *blockGradWeight = gradWeight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;
    float *blockWeight = weight + outputIdx*nInputBlock*outputSize*inputSize + inputIdx*outputSize*inputSize;
    
    for (int j=0; j<outputSize; j++)
    {
      float *rowWeight = blockWeight + j*inputSize;  
      float *rowGradWeight = blockGradWeight + j*inputSize;    
      
      buffer[tx] = 0;
      for (int i=tx; i<inputSize; i+=i_step)
      {
        // update weights and zero grad weight
        float w = rowWeight[i];
        w -= rowGradWeight[i]*lr;
        rowGradWeight[i] = 0;
        // norm of row
        buffer[tx] += w*w;
        rowWeight[i] = w;
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
        // renormalize
        for (int i=tx; i<inputSize; i+=i_step)
        {
          rowWeight[i] *= norm;
        }
      }
    }
  }
}

//TODO update and zero bias


static int cunnx_BlockSparse_updateParameters(lua_State *L)
{ 
  float lr = (float)lua_tonumber(L, 2);
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  int outputSize = luaT_getfieldcheckint(L, 1, "outputSize");
  int nInputBlock = luaT_getfieldcheckint(L, 1, "nInputBlock");
  int nOutputBlock = luaT_getfieldcheckint(L, 1, "nOutputBlock");
  float maxnorm = (float)luaT_getfieldchecknumber(L, 1, "maxNorm");
  
  // nOutputBlock x nInputBlock x outputSize x inputSize
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  // nOutputBlock x outputSize
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  
  THCudaTensor *paramUpdateCuda = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "paramUpdateCuda", "torch.CudaTensor");
  THIntTensor *paramUpdateHost = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "paramUpdateHost", "torch.IntTensor");
  
  int n = 0;
  
  /* table is in the stack at index -1 */
  lua_getfield(L, 1, "updates");
  lua_pushnil(L);  /* first key */
  n = 0;
  while (lua_next(L, -2) != 0) 
  {
    /* 'key' (at index -2) and 'value' (at index -1) */
    int inputIdx = (int)lua_tonumber(L, -2);
    lua_pushnil(L);  /* first key */
    while (lua_next(L, -2) != 0) 
    {
      /* uses 'key' (at index -2) and 'value' (at index -1) */
      int outputIdx = (int)lua_tonumber(L, -2);
      float scale = (float)lua_tonumber(L, -1);
      /* removes 'value'; keeps 'key' for next iteration */
      lua_pop(L, 1);
      n += 1;
    }
    /* removes 'value'; keeps 'key' for next iteration */
    lua_pop(L, 1);
  }
  
  if (n == 0) 
    return 0;
  
  THIntTensor_resize2d(paramUpdateHost, 2, n);
  THCudaTensor_resize2d(paramUpdateCuda, 2, n);
  
  /* table is in the stack at index -1 */
  lua_getfield(L, 1, "updates");
  lua_pushnil(L);  /* first key */
  n = 0;
  while (lua_next(L, -2) != 0) 
  {
    /* 'key' (at index -2) and 'value' (at index -1) */
    int inputIdx = (int)lua_tonumber(L, -2);
    lua_pushnil(L);  /* first key */
    while (lua_next(L, -2) != 0) 
    {
      /* uses 'key' (at index -2) and 'value' (at index -1) */
      int outputIdx = (int)lua_tonumber(L, -2);
      float scale = (float)lua_tonumber(L, -1);
      /* removes 'value'; keeps 'key' for next iteration */
      lua_pop(L, 1);
      // add block to paramUpdate tensor
      THIntTensor_set2d(paramUpdateHost, 0, n, inputIdx-1);
      THIntTensor_set2d(paramUpdateHost, 1, n, outputIdx-1);
      
      n += 1;
    }
    /* removes 'value'; keeps 'key' for next iteration */
    lua_pop(L, 1);
  }
  
  // send block indices to device
  THCudaTensor_copyInt(paramUpdateCuda, paramUpdateHost);
  
  /* call cudakernel */
  dim3 blocks(min(65535,n)); // each cuda block is a param block to be updated
  dim3 threads(BLOCKSPARSE_THREADS);
  cunnx_BlockSparse_updateParameters_kernel<<<blocks,threads>>>(
    THCudaTensor_data(weight), THCudaTensor_data(bias), 
    THCudaTensor_data(gradWeight), THCudaTensor_data(gradBias), 
    THCudaTensor_data(paramUpdateCuda), lr, maxnorm, n,
    inputSize, outputSize, nInputBlock, nOutputBlock
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
  {"BlockSparse_updateParameters", cunnx_BlockSparse_updateParameters},
  {NULL, NULL}
};

static void cunnx_BlockSparse_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunnx_BlockSparse__, "nn");
  lua_pop(L,1);
}
