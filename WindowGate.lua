local WindowGate, parent = torch.class('nn.WindowGate', 'nn.Module')
------------------------------------------------------------------------
--[[ WindowGate ]]--
-- Returns a table of {scales, indices}
-- TODO add gaussian jumps for robustness
------------------------------------------------------------------------

function WindowGate:__init(outputWindowSize, outputSize, softmax)
   parent.__init(self)
   self.outputWindowSize = outputWindowSize
   self.outputSize = outputSize
   self.softmax = softmax or true
   
   self.outputIndice = torch.LongTensor()
   self.inputIndice = torch.LongTensor()
   self._outputIndice = torch.Tensor()
   self._inputIndice = torch.Tensor()
   self._output = torch.Tensor()
   self._outputWindow = torch.Tensor()
   self.sum = torch.Tensor()
   self.centroid = torch.Tensor()
   self.output = {self._output, self.outputIndice}
   self.batchSize = 0
end

function WindowGate:updateOutput(input)
   assert(input:dim() == 2, "Only works with matrices")
   if self.batchSize ~= input:size(1) then
      self.inputWindowSize = input:size(2)*self.outputWindowSize/self.outputSize
      assert(self.inputWindowSize == math.ceil(self.inputWindowSize), "inputWindowSize should be an integer")
      assert(self.inputWindowSize > 3, "windowSize is too small")
      self.windowStride = self.outputWindowSize/self.inputWindowSize
      assert(self.windowStride == math.ceil(self.windowStride), "windowStride should be an integer")
      self.range = torch.repeatTensor(torch.range(1,input:size(2)):typeAs(input),input:size(1),1)
      self.batchSize = input:size(1)
      self._output:resize(self.batchSize, self.outputWindowSize)
      self.inputIndice:resize(self.batchSize)
      self.outputIndice:resize(self.batchSize)
   end
   self.sum:cmul(self.range, input)
   -- get coordinate of centoid
   if self.softmax then
      self.centroid:sum(self.sum, 2)
   else
      self.centroid:mean(self.sum, 2)
   end
   -- make centroids a number between 0 and 1
   self.centroid:div(input:size(2))
   self._inputIndice:mul(self.centroid, input:size(2)):add(-self.inputWindowSize*0.5)
   self.inputIndice:copy(self._inputIndice)
   self._outputIndice:mul(self.centroid, self.outputSize):add(-self.outputWindowSize*0.5)
   self.outputIndice:copy(self._outputIndice)
   for i=1,self.batchSize do
      -- clip indices
      self.inputIndice[i] = math.min(self.inputIndice[i], input:size(2)-self.inputWindowSize)
      self.inputIndice[i] = math.max(self.inputIndice[i], 1)
      self.outputIndice[i] = math.min(self.outputIndice[i], self.outputSize-self.outputWindowSize)
      self.outputIndice[i] = math.max(self.outputIndice[i], 1)
      -- expand window
      local inputWindow = input[i]:narrow(1, self.inputIndice[i], self.inputWindowSize)
      local outputWindow = torch.repeatTensor(inputWindow, self.windowStride, 1)
      self._output[i]:copy(outputWindow:t():reshape(self.outputWindowSize))
   end
   return self.output
end

function WindowGate:updateGradInput(input, gradOutput)   
   
   return self.gradInput
end

function WindowGate:type(type)
   assert(type ~= 'torch.CudaType', "torch.CudaType not supported")
   self._output = self._output:type(type)
   self.gradInput = self.gradInput:type(type)
   self.output = {self._output, self.indice}
end

