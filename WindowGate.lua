local WindowGate, parent = torch.class('nn.WindowGate', 'nn.Module')
------------------------------------------------------------------------
--[[ WindowGate ]]--
-- Returns a table of {indices, scales}
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

function WindowGate:__init(outputWindowSize, outputSize, inputStdv, outputStdv, lr, noiseStdv)
   parent.__init(self)
   self.outputWindowSize = outputWindowSize
   self.outputSize = outputSize
   self.inputStdv = inputStdv or 2
   self.outputStdv = outputStdv or outputWindowSize/2
   self.noiseStdv = noiseStdv or outputWindowSize/(outputSize*2)
   self.train = true
   
   self.a = 1/(self.outputStdv*math.sqrt(2*math.pi))
   self.b = -1/(2*self.outputStdv*self.outputStdv)
   self.c = 1/(self.outputStdv*self.outputStdv)
   
   self.lr = lr or 0.1
   
   self.outputIndice = torch.LongTensor()
   self._output = torch.Tensor()
   self.centroid = torch.Tensor()
   self.normalizedCentroid = torch.Tensor()
   self.targetCentroid = torch.Tensor()
   self.noise = torch.randn(10000):mul(self.noiseStdv) -- ugly hack
   self.train = true
   self.error = torch.Tensor()
   self.output = {self.outputIndice, self._output}
   self.batchSize = 0
   
end

function WindowGate:updateOutput(input)
   assert(input:dim() == 2, "Only works with matrices")
   if self.batchSize ~= input:size(1) then
      self.batchSize = input:size(1)
      self.inputSize = input:size(2)
      self.noise:resize(self.batchSize)
      self.outputIndice:resize(self.batchSize)
      self.d = 1/(self.inputStdv*math.sqrt(2*math.pi))
      self.e = -1/(2*self.inputStdv*self.inputStdv)
   end
   input.nn.WindowGate_updateOutput(self, input)
   return self.output
end

function WindowGate:updateGradInput(input, gradOutputTable)   
   local gradOutput = gradOutputTable[2]
   local gradInput = input.nn.WindowGate_updateGradInput(self, input, gradOutput)
   return gradInput
end

function WindowGate:type(type)
   self._output = self._output:type(type)
   self.gradInput = self.gradInput:type(type)
   self.centroid = self.centroid:type(type)
   self.normalizedCentroid = self.normalizedCentroid:type(type)
   self.targetCentroid = self.targetCentroid:type(type)
   self.error = self.error:type(type)
   self.noise = self.noise:type(type)
   self.output = {self.outputIndice, self._output}
   if type == 'torch.CudaTensor' then
      self.outputIndiceCuda = torch.CudaTensor()
   end
end
