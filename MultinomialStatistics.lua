local MultinomialStatistics, parent = torch.class('nn.MultinomialStatistics', 'nn.Module')
------------------------------------------------------------------------
--[[ MultinomialStatistics ]]--
-- Gathers statistics from multinomial distributions (like SoftMax).
------------------------------------------------------------------------

function MultinomialStatistics:__init(nBatch)
   parent.__init(self)
   self.nBatch = nBatch or 10
   self.inputCache = torch.Tensor()
   self.prob = torch.Tensor()
   self.batchSize = 0
   self.startIdx = 1
end

function MultinomialStatistics:updateOutput(input)
   assert(input:dim() == 2, "Only works with 2D inputs (batches)")
   if self.batchSize ~= input:size(1) then
      self.inputCache:resize(input:size(1)*self.nBatch, input:size(2)):zero()
      self.batchSize = input:size(1)
      self.startIdx = 1
   end
   
   self.output = input
   -- keep track of previous batches of P(Y|X)
   self.inputCache:narrow(1, self.startIdx, input:size(1)):copy(input)
   -- P(X) is uniform for all X, i.e. P(X) = 1/c where c is a constant
   -- P(Y) = sum_x( P(Y|X)*P(X) )
   self.prob:sum(self.inputCache, 1):div(self.prob:sum())

   self.startIdx = self.startIdx + self.batchSize
   if self.startIdx > self.inputCache:size(1) then
      self.startIdx = 1
   end

   return self.output
end

function MultinomialStatistics:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
