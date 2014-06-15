local BlockSparse, parent = torch.class('nn.BlockSparse', 'nn.Module')
------------------------------------------------------------------------
--[[ BlockSparse ]]--
-- Use for Distributed Conditional Computation
-- Inputs and outputs are sparse
-- Weights are organized as a matrix of blocks.
------------------------------------------------------------------------

function BlockSparse:__init(nInputBlock, inputSize, nOutputBlock, outputSize)
   parent.__init(self)
   self.nInputBlock = nInputBlock
   self.nOutputBlock = nOutputBlock
   self.inputSize = inputSize
   self.outputSize = outputSize
   
   self.weight = torch.Tensor(nOutputBlock, nInputBlock, outputSize, inputSize)
   self.bias = torch.Tensor(nOutputBlock, outputSize)
   
   self.gradWeight = torch.Tensor(nOutputBlock, nInputBlock, outputSize, inputSize)
   self.gradBias = torch.Tensor(nOutputBlock, outputSize)
   
   self.updates = {}
   
   -- for dense inputs or outputs
   self.inputIndice = torch.Tensor()
   self.outputIndice = torch.Tensor()
   self.inputScale = torch.Tensor()
   self.outputScale = torch.Tensor()
   
   -- for backward
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
   end
   local output = input.nn.BlockSparse_updateOutput(
      self, input, inputIndice, outputIndice, inputScale, outputScale
   )
   return self:packOutput(output, outputIndice, outputScale)
end

function BlockSparse:updateGradInput(inputTable, gradOutputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   local gradOuput = self.unpackGradOutput(gradOutputTable)
   local gradInput, gradOutputScale = input.nn.BlockSparse_updateGradInput(
      self, input, inputIndice, outputIndice, inputScale, outputScale, gradOutput
   )
   self:packGradInput(outputIndice, gradInput, gradOutputScale)
   return self.gradInput
end

function BlockSparse:accGradParameters(inputTable, gradOutput, scale)
   local input, inputIndice, outputIndice, inputScale, outputScale = self:unpackInput(inputTable)
   local gradOuput = self.unpackGradOutput(gradOutputTable)
   scale = scale or 1
   --input.nn.BlockSparse_accGradParameters(self, input, inputIndice, outputIndice, inputScale, outputScale, gradOutput, scale)
end

function BlockSparse:unpackInput(inputTable)
   local input, inputIndice, outputIndice, inputScale, outputScale, innerTable
   -- 3 possible use cases
   if self.nInputBlock == 1 then
      -- Dense input, sparse output:
      -- The input and outputs are each a table of 3 tensors: {activation, {indices, scales}}
      input, innerTable = unpack(inputTable)
      outputIndice, outputScale = unpack(innerTable)
      inputIndice = self.inputIndice
      inputScale = self.inputScale
   elseif self.nOutputBlock == 1 then
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
      outputIndice, outputScale = unpack(inputTable[2])
   end 
   return input, inputIndice, outputIndice, inputScale, outputScale
end

function BlockSparse:unpackGradOutput(gradOutputTable)
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
   else
      -- Sparse input, sparse output:
      -- gradOutput is a multi-table of 3 tensors: {activation, {indices, scales}}
      gradOutput = gradOutputTable[1]
   end 
   return gradOutput
end

function BlockSparse:packGradInput(outputIndice, gradInput, gradOutputScale)
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

function BlockSparse:packOutput(output, outputIndice, outputScale)
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
   return gradOutput
end

-- when static is true, return parameters with static keys
-- i.e. keys that don't change from batch to batch
function BlockSparse:parameters(static)
   local params, grads = {}, {}
   local updated = false
   for outputIdx, updates in pairs(self.updates) do
      for inputIdx, scale in pairs(updates) do
         if static then
            local weightId = {inputIdx, outputIdx}
            params[weightId] = self.weight[outputIdx][inputIdx]
            grads[weightId] = self.gradWeight[outputIdx][inputIdx]
            params[outputIdx] = self.bias[outputIdx]
            grads[outputIdx] = self.gradBias[outputIdx]
         else
            table.insert(params, self.weight[outputIdx][inputIdx])
            table.insert(params, self.bias[outputIdx])
            table.insert(grads, self.gradWeight[outputIdx][inputIdx])
            table.insert(grads, self.gradBias[outputIdx])
         end
         updated = true
      end
   end
   if not updated then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   end
   return params, grads
end

function BlockSparse:updateParameters(learningRate, partial)
   local maxNorm = self.maxNorm
   if partial and self.output.nn.BlockSparse_updateParameters then
      self.output.nn.BlockSparse_updateParameters(self, learningRate)
   end
   local params, gradParams = self:parameters(partial)
   if params then
      for k,param in pairs(params) do
         param:add(-learningRate, gradParams[k])
         if param:dim() == 2 and maxNorm then
            param:renorm(2,1,maxNorm)
         end
      end
   end
end

function BlockSparse:getBlockParameters(inputIdx, outputIdx)
   local weight = self.weight[outputIdx][inputIdx]
   local bias = self.bias[outputIdx]
   local gradWeight = self.gradWeight[outputIdx][inputIdx]
   local gradBias = self.gradBias[outputIdx]
   return {weight, bias}, {gradWeight, gradBias}
end

function BlockSparse:zeroGradParameters(partial)
   local _,gradParams = self:parameters(partial)
   for k,gradParam in pairs(gradParams) do
      gradParam:zero()
   end
   self.updates = {}
end

function BlockSparse:type(type)
   if type and (type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' or type == 'torch.CudaTensor') then
      self.weight = self.weight:type(type)
      self.bias = self.bias:type(type)
      self.gradWeight = self.gradWeight:type(type)
      self.gradBias = self.gradBias:type(type)
      self.output = self.output:type(type)
      self.gradInput = self.gradInput:type(type)
      
      self.inputIndice = self.inputIndice:type(type)  
      self.outputIndice = self.outputIndice:type(type)  
      self.inputScale = self.inputScale:type(type)  
      self.outputScale = self.outputScale:type(type) 
      self.gradOutputScale = self.gradOutputScale:type(type) 
   end
   return self
end

-- generate a Clone that shares parameters and metadata 
-- without wasting memory
function BlockSparse:sharedClone()
   error"NotImplemented"
   return smt:share(self, 'weight', 'bias')
end

-- we do not need to accumulate parameters when sharing
BlockSparse.sharedAccUpdateGradParameters = BlockSparse.accUpdateGradParameters

