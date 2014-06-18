local NoisyReLU, parent = torch.class('nn.NoisyReLU','nn.Module')

function NoisyReLU:__init(sparsityFactor, threshold_lr, alpha, std)
   parent.__init(self)
   
   assert(0 <= sparsityFactor and sparsityFactor <= 1, 'sparsityFactor not within range [0, 1]')
   assert(0 < alpha and alpha < 1, 'alpha not within range (0, 1)') 
   self.sparsityFactor = sparsityFactor or 0.1
   self.threshold_lr = threshold_lr or 0.1
   
   -- std for the noise, default is no noise
   self.std = std or 0
   
   -- larger alpha means putting more weights on contemporary value 
   -- when calculating the moving average mean
   self.alpha = alpha or 0.01
   self.first_batch = true 
   
   self.threshold = torch.Tensor()
   self.mean_sparsity = torch.Tensor()
   self.val = 0
end

function NoisyReLU:updateOutput(input)
   
   noise = torch.zeros(input:size())
   -- noise is switch on during training 
   if self.std > 0 then
      noise = noise:normal(0, self.std)
   end
   
   if self.first_batch then
      self.threshold = torch.zeros(input:size(2))
   end

   input = input + noise
   
   self.output:resizeAs(input)
   for i=1,input:size(1) do 
      self.output[i] = input[i]:gt(self.threshold)
      self.output[i]:cmul(input[i])
   end
   
   -- find the training sparsity of a batch output
   sparsity = torch.gt(self.output > 0):sum(2):type('torch.FloatTensor'):div(self.output:size(2))
   sparsity:resize(self.output:size(2))
   
   -- recalculate mean sparsity, using exponential moving average   
   if self.first_batch then
      self.mean_sparsity = sparsity
      self.first_batch = false
   else
      self.mean_sparsity = self.alpha * sparsity + (1 - self.alpha) * self.mean_sparsity
   end
   
   -- update threshold, raise the threshold if the training activeness is larger than the desired activeness
   self.threshold = self.threshold + self.threshold_lr * (self.mean_sparsity - self.sparsityFactor)
   
   return self.output
end

function NoisyReLU:updateGradInput(input, gradOutput)
   input.nn.NoisyReLU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

