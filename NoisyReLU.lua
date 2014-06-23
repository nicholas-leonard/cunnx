local NoisyReLU, parent = torch.class('nn.NoisyReLU','nn.Module')

function NoisyReLU:__init(sparsityFactor, threshold_lr, alpha_range, std)

   -- Params
   -- sparsityFactor: the micro sparsity of signals through each neuron
   -- threshold_lr: the learning rate of learning the optimum threshold for each neuron
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
end


function NoisyReLU:updateOutput(input)
   
   if self.threshold:nElement() == 0 then
      self.threshold = torch.zeros(input:size(2))
   end
   
   local noise = torch.zeros(input:size())
   
   -- setting noise
   if self.std > 0 then
      noise = noise:normal(0, self.std)
   end
  
   input = input + noise 
     
   -- check if a neuron is active
   local activated = torch.Tensor():resizeAs(input)
   for i=1,input:size(1) do 
      activated[i] = torch.gt(input[i], self.threshold)
   end
   
   self.output = torch.cmul(activated, input)

   -- find the activeness of a neuron in each batch
   local sparsity = activated:sum(1):div(self.output:size(1))
   sparsity:resize(self.output:size(2))

   -- recalculate mean sparsity, using exponential moving average   
   if self.mean_sparsity:nElement() == 0 then
      self.mean_sparsity = sparsity
   else
   
      if self.alpha - self.decrement < self.alpha_range[3] then
         self.alpha = self.alpha_range[3]
      else
         self.alpha = self.alpha - self.decrement
      end

      self.mean_sparsity = (sparsity * self.alpha
                            + self.mean_sparsity
                            * (1 - self.alpha))
   end
   
   return self.output
end


function NoisyReLU:updateGradInput(input, gradOutput)

   local activated = torch.Tensor():resizeAs(input)
   for i=1,input:size(1) do 
      activated[i] = torch.gt(input[i], self.threshold)
   end
   
   self.gradInput = torch.cmul(gradOutput, activated)
   
   -- update threshold, raise the threshold if the training 
   -- activeness is larger than the desired activeness
   self.threshold = ((self.mean_sparsity - self.sparsityFactor) * self.threshold_lr 
                      + self.threshold)

   return self.gradInput
end

