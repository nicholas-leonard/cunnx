local BlockMixture, parent = torch.class('nn.BlockMixture', 'nn.Module')
------------------------------------------------------------------------
--[[ BlockMixture ]]--
-- n > 1 BlockSparse Modules gated by a one gater with 
-- n - 1 output spaces, one for each of the hidden layers
------------------------------------------------------------------------

function BlockMixture:__init(experts, gater, expertScale, gaterScale)
   parent.__init(self)
   self.gater = gater
   assert(#experts > 1, "need at least 2 experts")
   self.experts = experts
   self.expertScale = expertScale or 1
   self.gaterScale = gaterScale or 1
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
   for i=1,#self.experts - 1 do
      self.expertInputs[i+1] = {self.experts[i]:updateOutput(self.expertInputs[i]), self.gaterOutputs[i+1]}
   end
   self.output = self.experts[#self.experts]:updateOutput(self.expertInputs[#self.expertInputs][1])
   
   return self.output
end

function BlockMixture:updateGradInput(input, gradOutput)
   self.expertGradInputs = {}
   self.expertGradInputs[#self.experts] = self.experts[#self.experts]:updateGradInput(self.expertInputs[#self.expertInputs][1], gradOutput)
   for i=#self.experts-1,1,-1 do
      self.expertGradInputs[i] = self.experts[i]:updateGradInput(self.expertInputs[i], self.expertGradInputs[i+1][1])
   end
   
   self.gaterGradOutputs = {}
   for i=1,#self.gaterOutputs do
      self.gaterGradOutputs[i] = self.expertGradInputs[i][2]
   end
   
   if #self.experts == 2 then
      self._gradInput = self.gater:updateGradInput(input, self.gaterGradOutputs[1])
   else
      self._gradInput = self.gater:updateGradInput(input, self.gaterGradOutputs)
   end
   
   self.gradInput:resizeAs(input)
   self.gradInput:copy(self.expertGradInputs[1][1])
   self.gradInput:add(self._gradInput)
   return self.gradInput
end

function BlockMixture:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.expertScale > 0 then
      self.experts[#self.experts]:accGradParameters(self.expertInputs[#self.expertInputs][1], gradOutput, scale)
      for i=#self.experts-1,1,-1 do
         self.experts[i]:accGradParameters(self.expertInputs[i], self.expertGradInputs[i+1][1], scale)
      end
   end
   
   if self.gaterScale > 0 then
      if #self.experts == 2 then
         self.gater:accGradParameters(input, self.gaterGradOutputs[1], scale)
      else
         self.gater:accGradParameters(input, self.gaterGradOutputs, scale)
      end
   end
end

function BlockMixture:accUpdateGradParameters(input, gradOutput, lr)
   if self.expertScale > 0 then
      self.experts[#self.experts]:accUpdateGradParameters(self.expertInputs[#self.expertInputs][1], gradOutput, lr)
      for i=#self.experts-1,1,-1 do
         self.experts[i]:accUpdateGradParameters(self.expertInputs[i], self.expertGradInputs[i+1][1], lr*self.expertScale)
      end
   end
   
   if self.gaterScale > 0 then
      if #self.experts == 2 then
         self.gater:accUpdateGradParameters(input, self.gaterGradOutputs[1], lr*self.gaterScale)
      else
         self.gater:accUpdateGradParameters(input, self.gaterGradOutputs, lr*self.gaterScale)
      end
   end
end

function BlockMixture:zeroGradParameters()
   for i,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function BlockMixture:updateParameters(learningRate)
   if self.expertScale > 0 then
      for i,expert in ipairs(self.experts) do
         expert:updateParameters(learningRate*self.expertScale)
      end
   end
   if self.gaterScale > 0 then
      self.gater:updateParameters(learningRate*self.gaterScale)
   end
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
