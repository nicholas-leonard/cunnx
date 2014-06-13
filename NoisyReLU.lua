local NoisyReLU, parent = torch.class('nn.NoisyReLU','nn.Module')

function NoisyReLU:__init(sparsityFactor)
   parent.__init(self)
   self.sparsityFactor = sparsityFactor or 0.1
   
   self.threshold = 0
   self.val = 0
end

function NoisyReLU:updateOutput(input)
   input.nn.NoisyReLU_updateOutput(self, input)
   return self.output
end

function NoisyReLU:updateGradInput(input, gradOutput)
   input.nn.NoisyReLU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

