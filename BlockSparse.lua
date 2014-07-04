local BlockSparse, parent = torch.class('nn.BlockSparse', 'nn.Module')
------------------------------------------------------------------------
--[[ BlockSparse ]]--
-- Use for Distributed Conditional Computation
-- Inputs and outputs are sparse
-- Weights are organized as a matrix of blocks.
------------------------------------------------------------------------

-- 1. Dense input, sparse output:
-- Input : {activation, {outputIndices, outputScales}}
-- Output : {activation, {outputIndices, outputScales}}

-- 2. Sparse input, sparse output:
-- Input : {{activation, {inputIndices, inputScales}}, {outputIndices, outputScales}}
-- Output : {activation, {inputIndices, inputScales}}

-- 3. Sparse input, dense output:
-- Input : {activation, {inputIndice, inputScale}}
-- Output : tensor of activations.

function BlockSparse:__init(nInputBlock, inputSize, nOutputBlock, outputSize, accUpdate)
   parent.__init(self)
   self.nInputBlock = nInputBlock
   self.nOutputBlock = nOutputBlock
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.accUpdate = accUpdate or false
   
   self.weight = torch.Tensor(nOutputBlock, nInputBlock, outputSize, inputSize)
   self.bias = torch.Tensor(nOutputBlock, outputSize)
   
   if not self.accUpdate then
      self.gradWeight = torch.Tensor(nOutputBlock, nInputBlock, outputSize, inputSize):zero()
      self.gradBias = torch.Tensor(nOutputBlock, outputSize):zero()
   end
   
   -- sqrt(inputWindowSize*outputWindowSize) smaller than this use 
   -- cublasSgemmBatched. 
   self.batchedGemmMax = 200
   
   -- for dense inputs or outputs
   self.inputIndice = torch.Tensor()
   self.outputIndice = torch.Tensor()
   self.inputScale = torch.Tensor()
   self.outputScale = torch.Tensor()
   
   -- for cuda
   self.inputIndiceHost = torch.LongTensor()
   self.outputIndiceHost = torch.LongTensor()
   
   self.inputHost = torch.CharTensor()
   self.weightHost = torch.CharTensor()
   self.outputHost = torch.CharTensor()
   
   -- etc
   self._output = torch.Tensor()
   self.gradOutputScale = torch.Tensor()
   self._gradInput = torch.Tensor()
   self.gradInput = {}

   self.batchSize = 0
   
   self:reset()
end

function BlockSparse:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.nInputBlock*0.1*self.inputSize)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function BlockSparse:updateOutput(inputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   if batchSize ~= input:size(1) then
      self.inputIndice:resize(input:size(1),1):fill(1)
      self.outputIndice:resize(input:size(1),1):fill(1)
      self.inputScale:resize(input:size(1),1):fill(1)
      self.outputScale:resize(input:size(1),1):fill(1)
      self.batchSize = input:size(1)
      self.inputWindowSize = inputIndice:size(2)
      self.outputWindowSize = outputIndice:size(2)
   end
   local output = input.nn.BlockSparse_updateOutput(
      self, input, inputIndice, outputIndice, inputScale, outputScale
   )
   self.output = self:packOutput(output, outputIndice, outputScale)
   return self.output
end

function BlockSparse:updateGradInput(inputTable, gradOutputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   local gradOutput = self:unpackGradOutput(gradOutputTable)
   local gradInput, gradOutputScale = input.nn.BlockSparse_updateGradInput(
      self, input, inputIndice, outputIndice, inputScale, outputScale, gradOutput
   )
   self:packGradInput(outputIndice, gradInput, gradOutputScale)
   return self.gradInput
end

function BlockSparse:accGradParameters(inputTable, gradOutputTable, scale)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   local gradOutput = self:unpackGradOutput(gradOutputTable)
   scale = scale or 1
   input.nn.BlockSparse_accGradParameters(
      self, input, inputIndice, outputIndice, inputScale, outputScale, gradOutput, scale
   )
end

function BlockSparse:getBlockParameters(inputIdx, outputIdx)
   local weight = self.weight[outputIdx][inputIdx]
   local bias = self.bias[outputIdx]
   local gradWeight = self.gradWeight[outputIdx][inputIdx]
   local gradBias = self.gradBias[outputIdx]
   return {weight, bias}, {gradWeight, gradBias}
end

function BlockSparse:type(type)
   if type and (type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' or type == 'torch.CudaTensor') then
      self.weight = self.weight:type(type)
      self.bias = self.bias:type(type)
      if not self.accUpdate then
         self.gradWeight = self.gradWeight:type(type)
         self.gradBias = self.gradBias:type(type)
      end
      self._output = self._output:type(type)
      self._gradInput = self._gradInput:type(type)
      
      self.inputIndice = self.inputIndice:type(type)  
      self.outputIndice = self.outputIndice:type(type)  
      self.inputScale = self.inputScale:type(type)  
      self.outputScale = self.outputScale:type(type) 
      self.gradOutputScale = self.gradOutputScale:type(type) 
      if type == 'torch.CudaTensor' then
         self.inputCuda = torch.CudaTensor()
         self.weightCuda = torch.CudaTensor()
         self.outputCuda = torch.CudaTensor()
         self.outputBatched = torch.CudaTensor()
         self.gradInputBatched = torch.CudaTensor()
         self._gradOutput = torch.CudaTensor()
      end
   end
   return self
end

-- generate a Clone that shares parameters and metadata 
-- without wasting memory
function BlockSparse:sharedClone()
   error"NotImplemented"
end

-- we do not need to accumulate parameters when sharing
BlockSparse.sharedAccUpdateGradParameters = BlockSparse.accUpdateGradParameters

function BlockSparse:unpackInput(inputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale, innerTable
   if self.nInputBlock == 1 then
      input, innerTable = unpack(inputTable)
      outputIndice, outputScale = unpack(innerTable)
      inputIndice = self.inputIndice
      inputScale = self.inputScale
   elseif self.nOutputBlock == 1 then
      input, innerTable = unpack(inputTable)
      inputIndice, inputScale = unpack(innerTable)
      outputIndice = self.outputIndice
      outputScale = self.outputScale
   else
      input, innerTable = unpack(inputTable[1])
      inputIndice, inputScale = unpack(innerTable)
      outputIndice, outputScale = unpack(inputTable[2])
   end 
   return input, inputIndice, outputIndice, inputScale, outputScale
end

function BlockSparse:unpackGradOutput(gradOutputTable)
   local gradOutput
   if self.nInputBlock == 1 then 
      gradOutput = gradOutputTable[1]
   elseif self.nOutputBlock == 1 then 
      gradOutput = gradOutputTable
   else
      gradOutput = gradOutputTable[1]
   end 
   return gradOutput
end

function BlockSparse:packGradInput(outputIndice, gradInput, gradOutputScale)
   local gradInputTable = self.gradInput
   if self.nInputBlock == 1 then
      gradInputTable[1] = gradInput
      gradInputTable[2] = {outputIndice, gradOutputScale}
   elseif self.nOutputBlock == 1 then
      gradInputTable[1] = gradInput
      gradInputTable[2] = {outputIndice, gradOutputScale}
   else
      gradInputTable[1] = {gradInput}
      gradInputTable[2] = {outputIndice, gradOutputScale}
   end 
end

function BlockSparse:packOutput(output, outputIndice, outputScale)
   local outputTable
   if self.nInputBlock == 1 then
      outputTable = {output, {outputIndice, outputScale}}
   elseif self.nOutputBlock == 1 then
      outputTable = output
   else
      outputTable = {output, {outputIndice, outputScale}}
   end 
   return outputTable
end
