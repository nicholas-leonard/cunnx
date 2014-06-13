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
   self.inputIndices = torch.Tensor()
   self.outputIndices = torch.Tensor()
   self.inputScales = torch.Tensor()
   self.outputScales = torch.Tensor()

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
   local input, inputIndices, outputIndices, inputScales, outputScales, innerTable
   -- 3 possible use cases
   if self.nInputBlock == 1 then
      -- Dense input, sparse output:
      -- The input and outputs are each a table of 3 tensors: {activation, {indices, scales}}
      input, innerTable = unpack(inputTable)
      outputIndices, outputScales = unpack(innerTable)
      inputIndices = self.inputIndices
      inputScales = self.inputScales
   elseif self.nOutputBlock == 1 then
      -- Sparse input, dense output:
      -- Input is a multi-table of 3 tensors: {activation, {indices, scales}}
      -- Output is a tensor of activations.
      input, innerTable = unpack(inputTable)
      inputIndices, inputScales = unpack(innerTable)
      outputIndices = self.outputIndices
      outputScales = self.outputScales
   else
      -- Sparse input, sparse output:
      -- Input is a multi-table of 5 tensors: {{activation, {indices, scales}}, {indices, scales}}
      -- Output is a multi-table of 3 tensors: {activation, {indices, scales}}
      input, innerTable = unpack(inputTable[1])
      inputIndices, inputScales = unpack(innerTable)
      outputIndices, outputScales = unpack(inputTable[2])
   end 
   
   if batchSize ~= input:size(1) then
      self.inputIndices:resize(input:size(1),1):fill(1)
      self.outputIndices:resize(input:size(1),1):fill(1)
      self.inputScales:resize(input:size(1),1):fill(1)
      self.outputScales:resize(input:size(1),1):fill(1)
      self.batchSize = input:size(1)
   end
   return input.nn.BlockSparse_updateOutput(self, input, inputIndices, outputIndices, inputScales, outputScales)
end

function BlockSparse:updateGradInput(inputTable, gradOutput)
   local input, input_indices, output_indices = unpack(inputTable)
   if self.gradInput then
      return input.nn.BlockSparse_updateGradInput(self, input, gradOutput, input.nn.BlockSparse_updateOutput(self, input, inputIndices, outputIndices, inputScales, outputScales))
   end
end

function BlockSparse:accGradParameters(inputTable, gradOutput, scale)
   local input, inputIndices, outputIndices, inputScales, outputScales = unpack(inputTable)
   scale = scale or 1
   input.nn.BlockSparse_accGradParameters(self, input, gradOutput, inputIndices, outputIndices, inputScales, outputScales, scale)
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
      
      self.inputIndices = self.inputIndices:type(type)  
      self.outputIndices = self.outputIndices:type(type)  
      self.inputScales = self.inputScales:type(type)  
      self.outputScales = self.outputScales:type(type)  
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

