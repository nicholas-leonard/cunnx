local WindowSparse, parent = torch.class('nn.WindowSparse', 'nn.Module')
------------------------------------------------------------------------
--[[ WindowSparse ]]--
-- Use for Distributed Conditional Computation
-- Inputs and outputs are sparse
-- Weights are organized as a matrix of blocks.
------------------------------------------------------------------------

function WindowSparse:__init(inputSize, outputSize, outputWindowSize, accUpdate)
   parent.__init(self)
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.outputWindowSize = outputWindowSize
   self.accUpdate = accUpdate or false
   
   self._output = torch.Tensor()
   self.output = {}
   
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   
   if not self.accUpdate then
      self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
      self.gradBias = torch.Tensor(outputSize):zero()
   end
   
   -- for cuda
   self.inputHost = torch.CharTensor()
   self.weightHost = torch.CharTensor()
   self.biasHost = torch.CharTensor()
   self.outputHost = torch.CharTensor()
   
   self.inputCuda = torch.CudaTensor()
   self.weightCuda = torch.CudaTensor()
   self.biasCuda = torch.CudaTensor()
   self.outputCuda = torch.CudaTensor()
   
   self.inputIndiceCuda = torch.CudaTensor()
   self.outputIndiceCuda = torch.CudaTensor()
   
   -- sqrt(inputWindowSize*outputWindowSize) smaller than this use 
   -- cublasSgemmBatched
   self.batchedGemmMax = 200
   
   -- for backward
   self.gradOutputScale = torch.Tensor()
   self._gradInput = torch.Tensor()
   self.gradInput = {}

   self.batchSize = 0
   
   self:reset()
end

function WindowSparse:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function WindowSparse:updateOutput(inputTable)
   local input, inputIndice, outputIndice = unpack(inputTable)
   if batchSize ~= input:size(1) then
      self.batchSize = input:size(1)
   end
   --print("windowsparse", input, inputIndice, outputIndice, inputScale, outputScale)
   self._output = input.nn.WindowSparse_updateOutput(self, input, inputIndice, outputIndice)
   --print("output", self._output, self:packOutput(self._output, outputIndice, outputScale))
   self.output[1] = self._output
   self.output[2] = outputIndice
   return self.output
end

function WindowSparse:updateGradInput(inputTable, gradOutputTable)
   local input, inputIndice, outputIndice = unpack(inputTable)
   local gradOutput = gradOutputTable[1]
   local gradInput = input.nn.WindowSparse_updateGradInput(self, input, inputIndice, outputIndice, gradOutput)
   self.gradInput[1] = gradInput
   return self.gradInput
end

function WindowSparse:accGradParameters(inputTable, gradOutputTable, scale)
   local input, inputIndice, outputIndice = unpack(inputTable)
   local gradOutput = gradOutputTable[1]
   scale = scale or 1
   input.nn.WindowSparse_accGradParameters(self, input, inputIndice, outputIndice, gradOutput, scale)
end

function WindowSparse:type(type)
   if type and (type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' or type == 'torch.CudaTensor') then
      self.weight = self.weight:type(type)
      self.bias = self.bias:type(type)
      if not self.accUpdate then
         self.gradWeight = self.gradWeight:type(type)
         self.gradBias = self.gradBias:type(type)
      end
      self._output = self._output:type(type)
      self._gradInput = self._gradInput:type(type)
   
      self.inputScale = self.inputScale:type(type)  
      self.outputScale = self.outputScale:type(type) 
      self.gradOutputScale = self.gradOutputScale:type(type) 
   end
   return self
end

-- generate a Clone that shares parameters and metadata 
-- without wasting memory
function WindowSparse:sharedClone()
   error"NotImplemented"
end

-- we do not need to accumulate parameters when sharing
WindowSparse.sharedAccUpdateGradParameters = WindowSparse.accUpdateGradParameters

