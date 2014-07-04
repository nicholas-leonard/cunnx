local WindowGate2, parent = torch.class('nn.WindowGate2', 'nn.Module')
------------------------------------------------------------------------
--[[ WindowGate2 ]]--
-- Returns a table of {indices, scales}
-- Forward finds the centroid of the input (output of a softmax).
-- The centroid is used to position a window on the outputs.
-- Outputs an upscalled version of window.
-- Backward generates a gradient for gaussian parameter mu, which is 
-- then backwarded to the input.
------------------------------------------------------------------------

function WindowGate2:__init(outputWindowSize, outputSize, inputStdv, outputStdv, lr, noiseStdv)
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

function WindowGate2:updateOutput(input)
   assert(input:dim() == 2, "Only works with matrices")
   if self.batchSize ~= input:size(1) then
      self.batchSize = input:size(1)
      self.inputSize = input:size(2)
      
      self.inputWindowSize = input:size(2)*self.outputWindowSize/self.outputSize
      assert(self.inputWindowSize == math.ceil(self.inputWindowSize), "inputWindowSize should be an integer")
      assert(self.inputWindowSize > 3, "windowSize is too small")
      self.windowStride = self.outputWindowSize/self.inputWindowSize
      assert(self.windowStride == math.ceil(self.windowStride), "windowStride should be an integer")
      
      self.noise:resize(self.batchSize)
      self.outputIndice:resize(self.batchSize)
      self.d = 1/(self.inputStdv*math.sqrt(2*math.pi))
      self.e = -1/(2*self.inputStdv*self.inputStdv)
   end
   input.nn.WindowGate2_updateOutput(self, input)
   return self.output
end

function WindowGate2:updateGradInput(input, gradOutputTable)   
   local gradOutput = gradOutputTable[2]
   local gradInput = input.nn.WindowGate2_updateGradInput(self, input, gradOutput)
   return gradInput
end

function WindowGate2:type(type)
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
      self.inputIndiceCuda = torch.CudaTensor()
   end
end
