local NoisyReLU, parent = torch.class('nn.NoisyReLU','nn.Module')

function NoisyReLU:__init(sparsityFactor, threshold_lr, alpha)
   parent.__init(self)
   
   assert(0 <= sparsityFactor and sparsityFactor <= 1, 'sparsityFactor not within range [0, 1]')
   assert(0 < alpha and alpha < 1, 'alpha not within range (0, 1)') 
   self.sparsityFactor = sparsityFactor or 0.1
   self.threshold_lr = threshold_lr or 0.1
   
   -- larger alpha means putting more weights on contemporary value 
   -- when calculating the moving average
   self.alpha = alpha or 0.1
   self.first_batch = true 
   self.batchSize = 0
   self.threshold = torch.Tensor()
   self.mean_sparsity = torch.Tensor()
   self.sparsity = torch.Tensor()
   self.gt = torch.Tensor()
   
   self.val = 0
end

function NoisyReLU:updateOutput(input)
   assert(input:dim() == 2, "Expect 2D input (batch-mode)")
   if self.batchSize ~= input:size(1) then
      self.batchSize = input:size(1)
      self.threshold:resize(input:size(2))
      self.mean_sparsity:resize(input:size(2))
   end
   local output = input.nn.Threshold_updateOutput(self, input)
   
   -- find the training sparsity of a batch output
   self.gt:gt(output, 0)
   self.sparsity:mean(2)
   --sparsity:resize(self.output:size(2))
   
   -- recalculate mean sparsity, using exponential moving average   
   if self.first_batch then
      self.mean_sparsity = sparsity
      self.first_batch = false
   else
      self.mean_sparsity = self.alpha * sparsity + (1 - self.alpha) * self.mean_sparsity
   end
   
   -- update threshold, raise the threshold if the training sparsity is larger than the desired sparsity
   self.threshold = self.threshold + self.threshold_lr * (self.mean_sparsity - self.sparsityFactor)
   
   return self.output
end

function NoisyReLU:updateGradInput(input, gradOutput)
   input.nn.Threshold_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

