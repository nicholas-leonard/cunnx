local WindowGate, parent = torch.class('nn.WindowGate', 'nn.Module')
------------------------------------------------------------------------
--[[ WindowGate ]]--
-- Returns a table of {scales, indices}
-- Forward finds the centroid of the input (output of a softmax).
-- Centroid is then uses as mu (mean) to generate a gaussian blur
-- for the scales.
-- The centroid is also used to position a window on the outputs.
-- Backward generates a gradient for gaussian parameter mu, which is 
-- then backwarded to the input.
-- So in effect, this layer outputs a gaussian blur and training can 
-- learn to move it around.
-- TODO add gaussian jumps for robustness
------------------------------------------------------------------------

function WindowGate:__init(outputWindowSize, outputSize, inputStdv, outputStdv, lr)
   parent.__init(self)
   self.outputWindowSize = outputWindowSize
   self.outputSize = outputSize
   self.inputStdv = inputStdv or 2
   self.outputStdv = outputStdv or outputWindowSize/2
   
   self.a = 1/(self.outputStdv*math.sqrt(2*math.pi))
   self.b = -1/(2*self.outputStdv*self.outputStdv)
   self.c = 1/(self.outputStdv*self.outputStdv)
   
   self.lr = lr or 0.1
   
   self.outputIndice = torch.LongTensor()
   self._output = torch.Tensor()
   self.centroid = torch.Tensor()
   self.error = torch.Tensor()
   self.output = {self._output, self.outputIndice}
   self.batchSize = 0
   
end

function WindowGate:updateOutput(input)
   assert(input:dim() == 2, "Only works with matrices")
   if self.batchSize ~= input:size(1) then
      self.batchSize = input:size(1)
      self.inputSize = input:size(2)
      self.outputIndice:resize(self.batchSize)
      self.inputStdv = inputStdv or input:size(2)/2
      self.d = 1/(self.inputStdv*math.sqrt(2*math.pi))
      self.e = -1/(2*self.inputStdv*self.inputStdv)
   end
   input.nn.WindowGate_updateOutput(self, input)
   return self.output
end

function WindowGate:updateGradInput(input, gradOutputTable)   
   local gradOutput = gradOutputTable[1]
   return input.nn.WindowGate_updateGradInput(self, input, gradOutput)
end

function WindowGate:type(type)
   self._output = self._output:type(type)
   self.gradInput = self.gradInput:type(type)
   self.centroid = self.centroid:type(type)
   self.error = self.error:type(type)
   self.output = {self._output, self.outputIndice}
   if type == 'torch.CudaTensor' then
      self.outputIndiceCuda = torch.CudaTensor()
   end
end
