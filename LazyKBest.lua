local LazyKBest, parent = torch.class('nn.LazyKBest', 'nn.Module')
------------------------------------------------------------------------
--[[ LazyKBest ]]--
-- For example, divides the input into k sub-arrays and takes the 
-- max value of each. Allowed value for k are, 1, 2, 4, 8, 16 and 32.
-- Returns a table of the k-best {indices, inputs}
-- Used with BlockSparse instead of nn.Sort
------------------------------------------------------------------------

function LazyKBest:__init(k)
   parent.__init(self)   
   self.k = k
   self._indice = torch.LongTensor()
   self._output = torch.Tensor()
   self.output = {self._indice, self._output}
end

function LazyKBest:updateOutput(input)
   assert(input:dim() == 2, "Only works with matrices")
   input.nn.LazyKBest_updateOutput(self, input)
   return self.output
end

function LazyKBest:updateGradInput(input, gradOutput)
   input.nn.LazyKBest_updateGradInput(self, input, self._indice, gradOutput[2])
   return self.gradInput
end

function LazyKBest:type(type)
   self.gradInput = self.gradInput:type(type)
   self._output = self._output:type(type)
   if (type == 'torch.CudaTensor') then
      self._indice = self._indice:type(type)
   end
   self.output = {self._indice, self._output}
end


