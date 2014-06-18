local WindowSparse, parent = torch.class('nn.WindowSparse', 'nn.Module')
------------------------------------------------------------------------
--[[ WindowSparse ]]--
-- Use for Distributed Conditional Computation
-- Inputs and outputs are sparse
-- Weights are organized as a matrix of blocks.
------------------------------------------------------------------------

function WindowSparse:__init(inputSize, outputSize, maxNorm)
   parent.__init(self)
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.maxNorm = maxNorm or 1
   
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   
   self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
   self.gradBias = torch.Tensor(outputSize):zero()
   
   -- for dense inputs or outputs
   self.inputIndice = torch.LongTensor()
   self.outputIndice = torch.LongTensor()
   self.inputScale = torch.Tensor()
   self.outputScale = torch.Tensor()
   
   -- for backward
   self.gradOutputScale = torch.Tensor()
   self._gradInput = torch.Tensor()
   self.gradInput = {}

   self.batchSize = 0
   
   self:reset()
end

function WindowSparse:updateOutput(inputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   if batchSize ~= input:size(1) then
      self.inputIndice:resize(input:size(1),1):fill(1)
      self.outputIndice:resize(input:size(1),1):fill(1)
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
   local gradInput, gradOutputScale = input.nn.WindowSparse_updateGradInput(
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
   self.zeroed = false
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
   if self.nInputBlock == 1 then
      -- Dense input, sparse output:
      -- The input and outputs are each a table of 3 tensors: {activation, {indice, scales}}
      input, innerTable = unpack(inputTable)
      outputIndice, outputScale = unpack(innerTable)
      inputIndice = self.inputIndice
      inputScale = self.inputScale
   elseif self.nOutputBlock == 1 and not #inputTable == 2 then
      -- Sparse input, dense output:
      -- Input is a multi-table of 3 tensors: {activation, {inputIndice, inputScale}}
      -- Output is a tensor of activations.
      input, innerTable = unpack(inputTable)
      inputIndice, inputScale = unpack(innerTable)
      outputIndice = self.outputIndice
      outputScale = self.outputScale
   else
      -- Sparse input, sparse output:
      -- Input is a multi-table of 5 tensors: {{activation, {indices, scales}}, {indices, scales}}
      -- Output is a multi-table of 3 tensors: {activation, {indices, scales}}
      input, innerTable = unpack(inputTable[1])
      inputIndice, inputScale = unpack(innerTable)
      if self.nOutputBlock > 1 then
         outputIndice, outputScale = unpack(inputTable[2])
      else 
         -- for gaters
         outputIndice = self.outputIndice
         outputScale = self.outputScale
      end
   end 
   return input, inputIndice, outputIndice, inputScale, outputScale
end

function WindowSparse:unpackGradOutput(gradOutputTable)
   local gradOutput
   -- 3 possible use cases
   if self.nInputBlock == 1 then 
      -- Dense input, sparse output:
      -- gradOutput is a table of 3 tensors: {activation, {indices, scales}}
      gradOutput = gradOutputTable[1]
   elseif self.nOutputBlock == 1 then 
      -- Sparse input, dense output:
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
   -- 3 possible use cases
   if self.nInputBlock == 1 then
      -- Dense input, sparse output:
      -- Input is a table of 3 tensors: {activation, {indices, scales}}
      gradInputTable[1] = gradInput
      gradInputTable[2] = {outputIndice, gradOutputScale}
   elseif self.nOutputBlock == 1 then
      -- Sparse input, dense output:
      -- Input is a multi-table of 3 tensors: {activation, {indices, scales}}
      gradInputTable[1] = gradInput
      gradInputTable[2] = {outputIndice, gradOutputScale}
   else
      -- Sparse input, sparse output:
      -- Input is a multi-table of 5 tensors: {{activation, {indices, scales}}, {indices, scales}}
      gradInputTable[1] = {gradInput}
      gradInputTable[2] = {outputIndice, gradOutputScale}
   end 
end

function WindowSparse:packOutput(output, outputIndice, outputScale)
   local outputTable
   -- 3 possible use cases
   if self.nInputBlock == 1 then
      -- Dense input, sparse output:
      -- output is a table of 3 tensors: {activation, {indices, scales}}
      outputTable = {output, {outputIndice, outputScale}}
   elseif self.nOutputBlock == 1 then
      -- Sparse input, dense output:
      -- output is a tensor of activations.
      outputTable = output
   else
      -- Sparse input, sparse output:
      -- output is a multi-table of 3 tensors: {activation, {indices, scales}}
      outputTable = {output, {outputIndice, outputScale}}
   end 
   return outputTable
end
