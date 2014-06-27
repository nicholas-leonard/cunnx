local NoisyReLU, parent = torch.class('nn.NoisyReLU','nn.Module')

function NoisyReLU:__init(sparsityFactor, threshold_lr, alpha_range, std)

   -- Params
   -- sparsityFactor: the micro sparsity of signals through each neuron
   -- threshold_lr: the rate for learning the optimum threshold for each neuron
   --               so that the activeness of the neuron approaches sparsityFactor
   -- alpha_range: {start_weight, num_batches, final_weight} for setting the weight on the
   --              contemporary sparsity when calculating the mean sparsity over many batches. 
   --              For the start, it will place more weight on the contemporary, but as more
   --              epoch goes through, the weight on contemporary batch should decrease, so
   --              that mean_sparsity will be more stable.
  
   
   parent.__init(self)
   
   self.sparsityFactor = sparsityFactor or 0.1
   self.threshold_lr = threshold_lr or 0.1
   
   -- std for the noise
   self.std = std or 1
      
   -- larger alpha means putting more weights on contemporary value 
   -- when calculating the moving average mean
   self.alpha_range = alpha_range or {0.5, 1000, 0.02}

   assert(self.alpha_range[2] % 1 == 0 and self.alpha_range[2] > 0) -- is an int and > 0
   assert(self.alpha_range[1] >= self.alpha_range[3] and self.alpha_range[3] >= 0)

   self.alpha = alpha_range[1]
   self.decrement = (alpha_range[1] - alpha_range[3]) / alpha_range[2]   

   assert(0 <= self.sparsityFactor and self.sparsityFactor <= 1, 
         'sparsityFactor not within range [0, 1]')
   assert(0 < self.alpha and self.alpha < 1, 
         'alpha not within range (0, 1)')
   assert(self.std >= 0, 'std has be >= 0')
   
   self.threshold = torch.Tensor()
   self.mean_sparsity = torch.Tensor()
   self.noise = torch.Tensor()
   self.activated = torch.Tensor()
   self.sparsity = torch.Tensor()
   self.threshold_delta = torch.Tensor()
   
   self.batchSize = 0
end


function NoisyReLU:updateOutput(input)
   assert(input:dim() == 2, "Only works with 2D inputs (batch-mode)")
   if self.batchSize ~= input:size(1) then
      self.output:resizeAS(input)
      self.noise:resizeAs(input)
      self.threshold:resize(1, input:size(2)):zero()
      -- setting noise
      if self.std > 0 then
         self.noise:normal(0, self.std)
      end
      self.batchSize = input:size(1)
   end
  
   self.output:copy(input)
   self.output:add(noise)
     
   -- check if a neuron is active
   self.activated:gt(input, self.threshold:expandAs(input))
   
   self.output:cmul(self.activated)

   -- find the activeness of a neuron in each batch
   self.sparsity:mean(self.activated, 1)

   -- recalculate mean sparsity, using exponential moving average   
   if self.mean_sparsity:nElement() == 0 then
      self.mean_sparsity:resize(input:size(2))
      self.mean_sparsity:copy(sparsity)
   else
   
      if self.alpha - self.decrement < self.alpha_range[3] then
         self.alpha = self.alpha_range[3]
      else
         self.alpha = self.alpha - self.decrement
      end

      self.mean_sparsity:mul(1-self.alpha):add(self.alpha, sparsity)
   end
   
   return self.output
end


function NoisyReLU:updateGradInput(input, gradOutput)
   
   self.gradInput:copy(gradOutput):cmul(self.activated)
   
   -- update threshold, raise the threshold if the training 
   -- activeness is larger than the desired activeness
   self.threshold_delta:copy(self.mean_sparsity)
   self.threshold_delta:add(-self.sparsity_factor)
   self.threshold:add(self.threshold_lr, self.threshold_delta)
   return self.gradInput
end

