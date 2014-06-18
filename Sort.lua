local Sort, parent = torch.class('nn.Sort', 'nn.Module')
------------------------------------------------------------------------
--[[ Sort ]]--
-- Applies torch.sort along dimension dim to the input.
-- Returns a table of {sortedInputs, sortedIndices}
-- Used with BlockSparse
------------------------------------------------------------------------

function Sort:__init(dim, descending)
   parent.__init(self)
   self.dim = dim or 2
   self.notDim = (self.dim == 2) and 1 or 2
   self.descending = descending or false
   
   self.indice = torch.LongTensor()
   self._output = torch.Tensor()
   self.output = {self._output, self.indice}
end

function Sort:updateOutput(input)
   assert(input:dim() == 2, "Only works with matrices")
   self._output:sort(self.indice, input, self.dim, self.descending)
   return self.output
end

function Sort:updateGradInput(input, gradOutput)   
   local dim = self.notDim
   self.gradInput:resizeAs(input)
   for i=1,input:size(dim) do
      self.gradInput:select(dim, i):indexCopy(1, self.indice:select(dim, i), gradOutput[1]:select(dim, i))
   end
   return self.gradInput
end

function Sort:type(type)
   assert(type ~= 'torch.CudaType', "torch.CudaType not supported")
   self._output = self._output:type(type)
   self.gradInput = self.gradInput:type(type)
   self.output = {self._output, self.indice}
end


