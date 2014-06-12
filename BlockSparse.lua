local BlockSparse, parent = torch.class('nn.BlockSparse', 'nn.Module')
------------------------------------------------------------------------
--[[ BlockSparse ]]--
-- Use for Distributed Conditional Computation
-- Inputs and outputs are sparse
-- Weights are organized as a matrix of blocks.
------------------------------------------------------------------------

function BlockSparse:__init(nInputBlock, inputSize, nOutputBlock, outputSize, sparsityFactor)
   parent.__init(self)
   self.nInputBlock = nInputBlock
   self.nOutputBlock = nOutputBlock
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.sparsityFactor = sparsityFactor
   
   self.weight = torch.Tensor(nOutputBlock, nInputBlock, outputSize, inputSize)
   self.bias = torch.Tensor(nOutputBlock, outputSize)
   
   self.gradWeight = torch.Tensor(nOutputBlock, nInputBlock, outputSize, inputSize)
   self.gradBias = torch.Tensor(nOutputBlock, outputSize)
   
   self.updates = {}

   self.batchSize = 0
   
   self:reset()
end

function BlockSparse:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.nInputBlock*sparsityFactor*self.inputSize)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function BlockSparse:updateOutput(inputTable)
   local input, input_indices, output_indices = unpack(inputTable)
   return input.nn.BlockSparse_updateOutput(self, input, input_indices, output_indices)
end

function BlockSparse:updateGradInput(inputTable, gradOutput)
   local input, input_indices, output_indices = unpack(inputTable)
   if self.gradInput then
      return input.nn.BlockSparse_updateGradInput(self, input, gradOutput, input_indices, output_indces)
   end
end

function BlockSparse:accGradParameters(inputTable, gradOutput, scale)
   local input, input_indices, output_indices = unpack(inputTable)
   scale = scale or 1
   input.nn.BlockSparse_accGradParameters(self, input, gradOutput, input_indices, output_indces, scale)
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
            table.insert(grads, self.gradBias:[outputIdx])
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
      self.batchSize = 0 --so that buffers are resized
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

