local TypeDecorator, parent = torch.class('nn.TypeDecorator', 'nn.Module')

function TypeDecorator:__init(module)
   parent.__init(self)
   self.module = module
end

function TypeDecorator:type(type)
   -- its a fixed type, so do nothing (the reason for this decorator)
   return
end

function TypeDecorator:updateOutput(input)
   self.output = self.module:updateOutput(input)
   return self.output
end

function TypeDecorator:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function TypeDecorator:accGradParameters(input, gradOutput, scale)
   return self.module:accGradParameters(input, gradOutput, scale)
end

function TypeDecorator:accUpdateGradParameters(input, gradOutput, lr)
   return self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function TypeDecorator:sharedAccUpdateGradParameters(input, gradOutput, lr)
   return self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function TypeDecorator:parameters()
   return self.module:parameters()
end

function TypeDecorator:zeroGradParameters()
   self.module:zeroGradParameters()
end

function TypeDecorator:updateParameters(learningRate)
   self.module:updateParameters(learningRate)
end

function TypeDecorator:share(mlp, ...)
   self.module:share(mlp, ...)
end

function TypeDecorator:reset()
   self.module:reset()
end

function TypeDecorator:getParameters()
   return self.module:getParameters()
end

function TypeDecorator:__call__(input, gradOutput)
   return self.module:__call__(input, gradOutput)
end
