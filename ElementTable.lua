local ElementTable, parent = torch.class('nn.ElementTable', 'nn.Module')

function ElementTable:__init(index)
   parent.__init(self)
   self.index = index
   self.gradInput = {}
end

function ElementTable:updateOutput(input)
   self.output:set(input[self.index])
   return self.output
end

function ElementTable:updateGradInput(input, gradOutput)
   self.gradInput[index] = gradOutput
   return self.gradInput
end
