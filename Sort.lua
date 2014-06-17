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
   self.notDim = (dim == 2) and 1 or 2
   self.descending = descending]
   
   self.indice = torch.LongTensor()
   self._output = torch.Tensor()
   self.output = {}
end

function Sort:updateOutput(input)
   assert(self.input:dim() == 2, "Only works with matrices")
   self._output:sort(self.indice, self.input, self.dim, self.descending)
   return self.output
end

function Sort:updateGradInput(input, gradOutput)
   local dim = self.notDim
   self.gradInput:resizeAs(input)
   for i=1,input:size(1) do
      self.gradInput:select(dim, i):index(gradOutput:select(dim, i), 1, self.indice:select(dim, i))
   end
   return self.gradInput
end


