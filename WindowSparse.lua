local WindowSparse, parent = torch.class('nn.WindowSparse', 'nn.Module')
------------------------------------------------------------------------
--[[ WindowSparse ]]--
-- Use for Distributed Conditional Computation
-- Inputs and outputs are sparse
-- Weights are organized as a matrix of blocks.
------------------------------------------------------------------------

-- 3 modes
-- Dense input, sparse output:
-- The input and outputs are each a table of 3 tensors: {activation, {indice, scales}}
WindowSparse.DENSE_SPARSE = 1
-- Sparse input, dense output:
-- Input is a multi-table of 3 tensors: {activation, {inputIndice, inputScale}}
-- Output is a tensor of activations.
WindowSparse.SPARSE_DENSE = 2
-- Sparse input, sparse output:
-- Input is a multi-table of 5 tensors: {{activation, {indices, scales}}, {indices, scales}}
-- Output is a multi-table of 3 tensors: {activation, {indices, scales}}
WindowSparse.SPARSE_SPARSE = 3

function WindowSparse:__init(inputSize, outputSize, mode, maxNorm)
   parent.__init(self)
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.maxNorm = maxNorm or 1
   self.mode = mode or self.SPARSE_SPARSE
   
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   
   self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
   self.gradBias = torch.Tensor(outputSize):zero()
   
   -- for dense inputs or outputs
   self.inputIndice = torch.LongTensor()
   self.outputIndice = torch.LongTensor()
   self.inputScale = torch.Tensor()
   self.outputScale = torch.Tensor()
   
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
   
   -- used for cmul(outputScale, output)
   self.cmultable = nn.CMulTable()

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
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   if batchSize ~= input:size(1) then
      self.inputIndice:resize(input:size(1)):fill(1)
      self.outputIndice:resize(input:size(1)):fill(1)
      self.inputScale:resize(input:size(1),1):fill(1)
      self.outputScale:resize(input:size(1),1):fill(1)
      self.batchSize = input:size(1)
   end
   local output = input.nn.WindowSparse_updateOutput(
      self, input, inputIndice, outputIndice, inputScale, outputScale
   )
   return self:packOutput(output, outputIndice, outputScale)
end

function WindowSparse:updateGradInput(inputTable, gradOutputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   local gradOutput = self:unpackGradOutput(gradOutputTable)
   local gradInput = input.nn.WindowSparse_updateGradInput(
      self, input, inputIndice, outputIndice, inputScale, outputScale, gradOutput
   )
   self:packGradInput(outputIndice, gradInput, gradOutputScale)
   return self.gradInput
end

function WindowSparse:accGradParameters(inputTable, gradOutputTable, scale)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   local gradOutput = self:unpackGradOutput(gradOutputTable)
   scale = scale or 1
   input.nn.WindowSparse_accGradParameters(
      self, input, inputIndice, outputIndice, inputScale, outputScale, gradOutput, scale
   )
end

function WindowSparse:type(type)
   if type and (type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' or type == 'torch.CudaTensor') then
      self.weight = self.weight:type(type)
      self.bias = self.bias:type(type)
      self.gradWeight = self.gradWeight:type(type)
      self.gradBias = self.gradBias:type(type)
      self.output = self.output:type(type)
      self._gradInput = self._gradInput:type(type)
   
      self.inputScale = self.inputScale:type(type)  
      self.outputScale = self.outputScale:type(type) 
      self.gradOutputScale = self.gradOutputScale:type(type) 
      self.cmultable:type(type)
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

function WindowSparse:unpackInput(inputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale, innerTable
   -- 3 possible use cases
   if self.mode == self.DENSE_SPARSE then
      input, innerTable = unpack(inputTable)
      outputIndice, outputScale = unpack(innerTable)
      inputIndice = self.inputIndice
      inputScale = self.inputScale
   elseif self.mode == self.SPARSE_DENSE and not #inputTable == 2 then
      input, innerTable = unpack(inputTable)
      inputIndice, inputScale = unpack(innerTable)
      outputIndice = self.outputIndice
      outputScale = self.outputScale
   else
      input, innerTable = unpack(inputTable[1])
      inputIndice, inputScale = unpack(innerTable)
      if self.mode == self.SPARSE_DENSE then
         -- for gaters
         outputIndice = self.outputIndice
         outputScale = self.outputScale
      else 
         outputIndice, outputScale = unpack(inputTable[2])
      end
   end 
   return input, inputIndice, outputIndice, inputScale, outputScale
end

function WindowSparse:unpackGradOutput(gradOutputTable)
   local gradOutput
   if self.mode == self.DENSE_SPARSE then 
      -- gradOutput is a table of 3 tensors: {activation, {indices, scales}}
      gradOutput = gradOutputTable[1]
   elseif self.nOutputBlock == 1 then 
      -- gradOutput is a tensor of activations.
      gradOutput = gradOutputTable
   else -- Sparse input, sparse output:
      -- gradOutput is a multi-table of 3 tensors: {activation, {indices, scales}}
      gradOutput = gradOutputTable[1]
   end 
   return gradOutput
end

function WindowSparse:packGradInput(outputIndice, gradInput, gradOutputScale)
   local gradInputTable = self.gradInput
   if self.mode == self.DENSE_SPARSE then
      -- Input is a table of 3 tensors: {activation, {indices, scales}}
      gradInputTable[1] = gradInput
      gradInputTable[2] = {outputIndice, gradOutputScale}
   elseif self.mode == self.SPARSE_DENSE then
      -- Input is a multi-table of 3 tensors: {activation, {indices, scales}}
      gradInputTable[1] = gradInput
      gradInputTable[2] = {outputIndice, gradOutputScale}
   else
      -- Input is a multi-table of 5 tensors: {{activation, {indices, scales}}, {indices, scales}}
      gradInputTable[1] = {gradInput}
      gradInputTable[2] = {outputIndice, gradOutputScale}
   end 
end

function WindowSparse:packOutput(output, outputIndice, outputScale)
   local outputTable
   -- 3 possible use cases
   if self.mode == self.DENSE_SPARSE then
      -- output is a table of 3 tensors: {activation, {indices, scales}}
      outputTable = {output, {outputIndice, outputScale}}
   elseif self.mode == self.SPARSE_DENSE then
      -- output is a tensor of activations.
      outputTable = output
   else
      -- output is a multi-table of 3 tensors: {activation, {indices, scales}}
      outputTable = {output, {outputIndice, outputScale}}
   end 
   return outputTable
end
