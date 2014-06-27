local BlockMixture, parent = torch.class('nn.BlockMixture', 'nn.Module')
------------------------------------------------------------------------
--[[ BlockMixture ]]--
-- n > 1 BlockSparse Modules gated by a one gater with 
-- n - 1 output spaces, one for each of the hidden layers
------------------------------------------------------------------------

function BlockMixture:__init(experts, gater)
   parent.__init(self)
   self.gater = gater
   assert(#experts > 1, "need at least 2 experts")
   self.experts = experts
   self.modules = {gater}
   for i, expert in ipairs(self.experts) do
      table.insert(self.modules, expert)
   end
  
   self.batchSize = 0
end

function BlockMixture:updateOutput(input)
   if #self.experts == 2 then
      self.gaterOutputs = {self.gater:updateOutput(input)}
   else
      self.gaterOutputs = self.gater:updateOutput(input)
   end
   
   self.expertInputs = {{input, self.gaterOutputs[1]}}
   for i, expert in ipairs(self.experts) do
      self.expertInputs[i+1] = {expert:updateOutput(self.expertInputs[1]), self.gaterOutputs[i+1]}
   end
   self.output = self.expertsInputs[#self.expertsInputs][1]
   
   return self.output
end

function BlockMixture:updateGradInput(input, gradOutput)
   self.expertGradOutputs = {}
   self.expertGradOutputs[#self.experts] = gradOutput
   for i=#self.experts,1,-1 do
      self.expertGradOutputs[i] = self.experts[i]:updateGradInput(self.expertInputs[i], self.expertGradOutputs[i+1])
   end
   
   self.gaterGradOutputs = {}
   for i=#self.gaterOutputs do
      self.gaterGradOutputs[i] = self.expertGradOutputs[i][2]
   end
   
   return self.gradInput

   
   self.expertGradInput = self.expert:updateGradInput(self.expertInput, {self.mixtureGradInput[1]})
   self.gaterGradInput = self.gater:updateGradInput(inputTable, self.mixtureGradInput[2])
   
   local gaterGradInput = self:unpackInput(self.gaterGradInput)
   self._gradInput:resizeAs(input)
   self._gradInput:copy(self.expertGradInput[1])
   self._gradInput:add(gaterGradInput)
   return self:packGradInput(self._gradInput)
end

function BlockMixture:accGradParameters(inputTable, gradOutputTable, scale)
   scale = scale or 1
   self.expert:accGradParameters(self.expertInput, {self.mixtureGradInput[2]}, scale)
   self.gater:accGradParameters(inputTable, self.mixtureGradInput[1], scale)
end

function BlockMixture:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   self.expert:accUpdateGradParameters(self.expertInput, {self.mixtureGradInput[2]}, lr)
   self.gater:accUpdateGradParameters(inputTable, self.mixtureGradInput[1], lr)
end

function BlockMixture:zeroGradParameters()
   self.expert:zeroGradParameters()
   self.gater:zeroGradParameters()
end

function BlockMixture:updateParameters(learningRate)
   self.expert:updateParameters(learningRate)
   self.gater:updateParameters(learningRate)
end

function BlockMixture:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function BlockMixture:parameters()
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

function BlockMixture:type(type)
   self.expert:type(type)
   self.gater:type(type)
   self.cmul:type(type)
   self._gradInput = self._gradInput:type(type)
   self.gradInput = (self.mode == self.DENSE_SPARSE) and self._gradInput or {}
end


function BlockMixture:unpackInput(inputTable)
   if self.mode == self.DENSE_SPARSE then
      return inputTable, self.inputIndice
   end 
   return unpack(inputTable)
end

function BlockMixture:unpackGradOutput(gradOutputTable)
   return gradOutputTable[1]
end

function BlockMixture:packGradInput(gradInput)
   if self.mode ~= self.DENSE_SPARSE then
      self.gradInput[1] = gradInput
   end
   return self.gradInput
end

function BlockMixture:packOutput(output, outputIndice)
   self.output[1] = output
   self.output[2] = outputIndice
   return self.output
end

