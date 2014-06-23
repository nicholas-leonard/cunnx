local WindowMixture, parent = torch.class('nn.WindowMixture', 'nn.Module')

-- 3 modes
-- Dense input, sparse output:
-- The input is a tensor of activations
-- outputs is a table of 2 tensors: {output, outputIndice}
WindowMixture.DENSE_SPARSE = 1
-- Sparse input, dense output:
-- Input is a table of 2 tensors: {input, inputIndice}
-- Output is a tensor of activations.
WindowMixture.SPARSE_DENSE = 2
-- Sparse input, sparse output:
-- Input is a table of 2 tensors: {input, inputIndice}
-- Output is a ttable of 2 tensors: {output, outputIndice}
WindowMixture.SPARSE_SPARSE = 3

function WindowMixture:__init(expert, gater, mode)
   parent.__init(self)
   self.gater = gater
   self.expert = expert
   self.mode = mode or self.SPARSE_SPARSE
   self.cmul = nn.CMulTable()
   self.modules = {gater, expert, cmul)
   self.output = self.SPARSE_DENSE and self.cmul.output or {}
   self._gradInput = torch.Tensor()
   self.gradInput = self.DENSE_SPARSE and self._gradInput or {}
   
   -- for dense inputs or outputs
   self.inputIndice = torch.LongTensor()
end

function WindowMixture:updateOutput(inputTable)
   if batchSize ~= input:size(1) then
      self.inputIndice:resize(input:size(1)):fill(1)
      self.batchSize = input:size(1)
   end
   local input, inputIndice = self:unpackInputTable(input)
   
   self.gaterOutput = self.gater:updateOutput(inputTable)
   
   self.expertInput = {input, inputIndice, self.gaterOutput[1]}
   self.expertOutput = self.expert:updateOutput(self.expertInput)
   
   self.mixtureInput = {self.gaterOutput[2], self.expertOutput[1]}
   self.mixtureOutput = self.cmul:updateOutput(self.mixtureInput)
   self:packOutput(self.mixtureOutput, self.gaterOutput[1])
   return self.output
end

function WindowMixture:updateGradInput(inputTable, gradOutputTable)
   local input, inputIndice = self:unpackInputTable(input)
   local gradOutput = self:unpackGradOutput(gradOutputTable)
   self.mixtureGradInput = self.cmul:updateGradInput(self.mixtureInput, gradOutput)
   
   self.expertGradInput = self.expert:updateGradInput(self.expertInput, {self.mixtureGradInput[2]})
   
   self.gaterGradInput = self.gater:updateGradInput(inputTable, self.mixtureGradInput[1])
   
   local gaterGradInput = self:unpackInputTable(self.gaterGradInput)
   self._gradInput:resizeAs(input)
   self._gradInput:copy(self.expertGradInput[1])
   self._gradInput:add(gaterGradInput)
   return self:packGradInput(self._gradInput)
end

function WindowMixture:accGradParameters(inputTable, gradOutputTable, scale)
   scale = scale or 1
   self.expertGradInput = self.expert:accGradParameters(self.expertInput, {self.mixtureGradInput[2]}, scale)
   self.gaterGradInput = self.gater:accGradParameters(inputTable, self.mixtureGradInput[1], scale)
end

function ConcatTable:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   scale = scale or 1
   self.expertGradInput = self.expert:accUpdateGradParameters(self.expertInput, {self.mixtureGradInput[2]}, lr)
   self.gaterGradInput = self.gater:accUpdateGradParameters(inputTable, self.mixtureGradInput[1], lr)
end

function WindowMixture:zeroGradParameters()
   self.expert:zeroGradParameters()
   self.gater:zeroGradParameters()
end

function WindowMixture:updateParameters(learningRate)
   self.expert:updateParameters(learningRate)
   self.gater:updateParameters(learningRate)
end

function WindowMixture:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function WindowMixture:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function WindowMixture:type(type)
   self.expert:type(type)
   self.gater:type(type)
   self.cmul:type(type)
   self.output = self.SPARSE_DENSE and self.cmul.output or {}
   self._gradInput = self._gradInput:type(type)
   self.gradInput = self.DENSE_SPARSE and self._gradInput or {}
end


function WindowMixture:unpackInput(inputTable)
   if self.mode == self.DENSE_SPARSE then
      return inputTable, self.inputIndice
   end 
   return unpack(inputTable)
end

function WindowMixture:unpackGradOutput(gradOutputTable)
   if self.mode ~= self.SPARSE_DENSE then 
      return gradOutputTable[1]
   end 
   return gradOutputTable
end

function WindowMixture:packGradInput(gradInput)
   if self.mode ~= self.DENSE_SPARSE then
      self.gradInput[1] = gradInput
   end
   return self.gradInput
end

function WindowMixture:packOutput(output, outputIndice)
   if self.mode ~= self.SPARSE_DENSE then
      -- output is a multi-table of 3 tensors: {activation, {indices, scales}}
      self.output[1] = output
      self.output[2] = outputIndice
   end
   return self.output
end

