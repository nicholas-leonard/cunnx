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
   self.expertGradOutputs[#self.experts + 1] = gradOutput
   for i=#self.experts,1,-1 do
      self.expertGradOutputs[i] = self.experts[i]:updateGradInput(self.expertInputs[i], self.expertGradOutputs[i+1])
   end
   
   self.gaterGradOutputs = {}
   for i=1,#self.gaterOutputs do
      self.gaterGradOutputs[i] = self.expertGradOutputs[i][2]
   end
   if #self.experts == 2 then
      self._gradInput = self.gater:updateGradInput(input, self.gaterGradOutputs[1])
   else
      self._gradInput = self.gater:updateGradInput(input, self.gaterGradOutputs)
   end
   
   self.gradInput:resizeAs(input)
   self.gradInput:copy(self.expertGradOutputs[1])
   self.gradInput:add(self._gradInput)
   return self.gradInput
end

function BlockMixture:accGradParameters(inputTable, gradOutputTable, scale)
   scale = scale or 1
   for i=#self.experts,1,-1 do
      self.experts[i]:accGradParameters(self.expertInputs[i], self.expertGradOutputs[i+1], scale)
   end
   
   if #self.experts == 2 then
      self.gater:accGradParameters(input, self.gaterGradOutputs[1], scale)
   else
      self.gater:accGradParameters(input, self.gaterGradOutputs, scale)
   end
end

function BlockMixture:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   for i=#self.experts,1,-1 do
      self.experts[i]:accUpdateGradParameters(self.expertInputs[i], self.expertGradOutputs[i+1], lr)
   end
   
   if #self.experts == 2 then
      self.gater:accUpdateGradParameters(input, self.gaterGradOutputs[1], lr)
   else
      self.gater:accUpdateGradParameters(input, self.gaterGradOutputs, lr)
   end
end

function BlockMixture:zeroGradParameters()
   for i,module = ipairs(self.modules) do
      module:zeroGradParameters()
   end
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
   for i,module in ipairs(self.modules) do
      module:type(type)
   end
   self.gradInput = self.gradInput:type(type)
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

