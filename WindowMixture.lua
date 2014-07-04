local WindowMixture, parent = torch.class('nn.WindowMixture', 'nn.Module')

-- 2 modes
-- Dense input, sparse output:
-- The input is a tensor of activations
-- outputs is a table of 2 tensors: {output, outputIndice}
WindowMixture.DENSE_SPARSE = 1
-- Sparse input, sparse output:
-- Input is a table of 2 tensors: {input, inputIndice}
-- Output is a ttable of 2 tensors: {output, outputIndice}
WindowMixture.SPARSE_SPARSE = 2

-- Note: SPARSE_DENSE is not supported (no gater is required), 
-- just use WindowSparse + ElementTable

function WindowMixture:__init(expert, gater, mode, mixture)
   parent.__init(self)
   self.gater = gater
   self.expert = expert
   self.mode = mode or self.SPARSE_SPARSE
   self.cmul = mixture or nn.CMulTable()
   self.modules = {gater, expert, cmul}
   self.output = {}
   self._gradInput = torch.Tensor()
   self.gradInput = (self.mode == self.DENSE_SPARSE) and self._gradInput or {}
   self.batchSize = 0
   
   -- for dense inputs or outputs
   self.inputIndice = torch.LongTensor()
end

function WindowMixture:updateOutput(inputTable)
   local input, inputIndice = self:unpackInput(inputTable)
   if self.batchSize ~= input:size(1) then
      self.inputIndice:resize(input:size(1)):fill(1)
      self.batchSize = input:size(1)
   end
   
   self.gaterOutput = self.gater:updateOutput(inputTable)
   
   self.expertInput = {input, inputIndice, self.gaterOutput[1]}
   self.expertOutput = self.expert:updateOutput(self.expertInput)
   
   self.mixtureInput = {self.expertOutput[1], self.gaterOutput[2]}
   self.mixtureOutput = self.cmul:updateOutput(self.mixtureInput)
   self:packOutput(self.mixtureOutput, self.gaterOutput[1])
   return self.output
end

function WindowMixture:updateGradInput(inputTable, gradOutputTable)
   local input, inputIndice = self:unpackInput(inputTable)
   local gradOutput = self:unpackGradOutput(gradOutputTable)
   self.mixtureGradInput = self.cmul:updateGradInput(self.mixtureInput, gradOutput)
   self.expertGradInput = self.expert:updateGradInput(self.expertInput, {self.mixtureGradInput[1]})
   self.gaterGradInput = self.gater:updateGradInput(inputTable, self.mixtureGradInput[2])
   
   local gaterGradInput = self:unpackInput(self.gaterGradInput)
   self._gradInput:resizeAs(input)
   self._gradInput:copy(self.expertGradInput[1])
   self._gradInput:add(gaterGradInput)
   return self:packGradInput(self._gradInput)
end

function WindowMixture:accGradParameters(inputTable, gradOutputTable, scale)
   scale = scale or 1
   self.expert:accGradParameters(self.expertInput, {self.mixtureGradInput[2]}, scale)
   self.gater:accGradParameters(inputTable, self.mixtureGradInput[1], scale)
end

function WindowMixture:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   self.expert:accUpdateGradParameters(self.expertInput, {self.mixtureGradInput[2]}, lr)
   self.gater:accUpdateGradParameters(inputTable, self.mixtureGradInput[1], lr)
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
   self._gradInput = self._gradInput:type(type)
   self.gradInput = (self.mode == self.DENSE_SPARSE) and self._gradInput or {}
end


function WindowMixture:unpackInput(inputTable)
   if self.mode == self.DENSE_SPARSE then
      return inputTable, self.inputIndice
   end 
   return unpack(inputTable)
end

function WindowMixture:unpackGradOutput(gradOutputTable)
   return gradOutputTable[1]
end

function WindowMixture:packGradInput(gradInput)
   if self.mode ~= self.DENSE_SPARSE then
      self.gradInput[1] = gradInput
   end
   return self.gradInput
end

function WindowMixture:packOutput(output, outputIndice)
   self.output[1] = output
   self.output[2] = outputIndice
   return self.output
end

