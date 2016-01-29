local cunnxtest = {}
local precision_forward = 1e-6
local precision_backward = 1e-6
local nloop = 50
local times = {}
local cunntestx = {}

--e.g. usage: th -lcunnx -e "nn.testcudax{'SoftMaxTree','BlockSparse'}"

local function round(a)
   if (a - math.floor(a)) >= 0.5 then
      return math.ceil(a)
   end
   return math.floor(a)
end 

local function blur(mean, stdv, size)
   local range = torch.range(1,size):float()
   local a = 1/(stdv*math.sqrt(2*math.pi))
   local b = -1/(2*stdv*stdv)
   return range:add(-mean):pow(2):mul(b):exp():mul(a)
end

function cunnxtest.SoftMaxTree()
   local input = torch.randn(120,100)
   local target = torch.repeatTensor(torch.IntTensor{20,23,27,10,8}, 24)
   local grad = torch.randn(120)
   local root_id = 29
   local hierarchy={
      [29]=torch.IntTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, 
      [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
      [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
      [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
      [8]=torch.IntTensor{24,25,26,27,28}
   }
   local smt = nn.SoftMaxTree(100, hierarchy, root_id)
   smt2 = smt:clone():cuda()

   local tm = {}
   local title = string.format('SoftMaxTree forward/backward ')
   times[title] = tm
   
   smt:zeroGradParameters()
   smt:forward{input, target}
   smt._multiBuffer:zero()
   local groundtruthF = smt:forward{input, target}:clone()
   local logsoftOutput = smt._multiBuffer:clone()
   local groundtruthB = smt:backward({input, target}, grad)[1]:clone()
   local linearOutput = smt._multiBuffer:clone()
   local gradWeight = smt.gradWeight:clone()
   local gradBias = smt.gradBias:clone()
   local weight = smt.weight:clone()
   local bias = smt.bias:clone()
   smt:zeroGradParameters()
   local a = torch.Timer()
   for i = 1,nloop do
      smt:forward{input, target}
      smt:backward({input, target}, grad)
      smt:updateParameters(0.1, true)
      smt:zeroGradParameters(true)
   end
   tm.cpu = a:time().real
   groundtruthF, groundtruthB = groundtruthF:clone(), groundtruthB:clone()
    
   input = input:cuda()
   target = target:float():cuda()
   grad = grad:cuda()
   smt2:zeroGradParameters()
   smt2:forward{input, target}
   smt2._multiBuffer:zero()
   local rescudaF = smt2:forward{input, target}:clone()
   local logsoftOutputCuda = smt2._multiBuffer:clone():float()
   local rescudaB = smt2:backward({input, target}, grad)[1]:clone()
   local linearOutputCuda = smt2._multiBuffer:clone():float()
   smt2._multiBuffer:zero()
   local gradWeightCuda = smt2.gradWeight:clone()
   local gradBiasCuda = smt2.gradBias:clone()
   local weightCuda = smt2.weight:clone()
   local biasCuda = smt2.bias:clone()
   smt2:zeroGradParameters()
   a:reset()
   for i = 1,nloop do
      smt2:forward{input, target}
      smt2:backward({input, target}, grad)
      smt2:updateParameters(0.1, true)
      smt2:zeroGradParameters(true)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real
   
   mytester:assertTensorEq(rescudaF:float(), groundtruthF, precision_forward, 'error on state (forward) ')
   mytester:assertTensorEq(logsoftOutput, logsoftOutputCuda, precision_forward, 'error on state (logsoftOutput) ')
   mytester:assertTensorEq(rescudaB:float(), groundtruthB, precision_backward, 'error on state (backward) ')
   mytester:assertTensorEq(linearOutput, linearOutputCuda, precision_forward, 'error on state (linearOutput) ')
   mytester:assertTensorEq(gradWeightCuda:float(), gradWeight, precision_backward*10, 'error on state (accGradParameters gradWeight) ')
   mytester:assertTensorEq(gradBiasCuda:float(), gradBias, precision_backward*10, 'error on state (accGradParameters gradBias) ')
   mytester:assertTensorEq(weightCuda:float(), weight, precision_backward, 'error on state (weight) ')
   mytester:assertTensorEq(biasCuda:float(), bias, precision_backward, 'error on state (bias) ')
   
   -- sharedClone
   if pcall(function() require "dpnn" end) then
      local smt3 = smt2:sharedClone()
      output = smt2:forward{input, target}
      output2 = smt3:forward{input, target}
      mytester:assertTensorEq(output:float(), output2:float(), 0.00001)
   end
   
   -- accUpdate
   local smt3 = nn.SoftMaxTree(100, hierarchy, root_id, true)
   smt3:cuda()
   smt3:zeroGradParameters()
   smt3.weight = smt2.weight:clone()
   smt3.bias = smt2.bias:clone()
   local output3 = smt3:forward{input, target}
   local output = smt2:forward{input, target}
   local gradInput3 = smt3:backwardUpdate({input, target}, grad, 0.1)[1]
   local gradInput = smt2:backwardUpdate({input, target}, grad, 0.1)[1]
   mytester:assertTensorEq(output3:float(), output:float(), 0.00001)
   mytester:assertTensorEq(gradInput3:float(), gradInput:float(), 0.00001)
   local parentId = 8
   local weight3, bias3 = unpack(smt3:getNodeParameters(parentId))
   local params = smt2:getNodeParameters(parentId)
   local weight, bias = unpack(params)
   mytester:assertTensorEq(weight3:float(), weight:float(), 0.00001)
   mytester:assertTensorEq(bias3:float(), bias:float(), 0.00001)
end

function cunnxtest.SoftMaxTree_issue24()
   local input = torch.randn(120,100):float()
   local targs = torch.IntTensor{3,4,5,6,7,8}
   local target = targs:index(1,torch.rand(120):mul(targs:size(1)):ceil():long())
   local root_id = 1
   local hierarchy={
       [1]=torch.IntTensor{2,7},
       [2]=torch.IntTensor{8,6,5,4,3},
       [7]=torch.IntTensor{9,10,11}
   }
   local smt = nn.SoftMaxTree(100,hierarchy,root_id):float()
   local tll = nn.TreeNLLCriterion():float()

   local inputGPU = input:cuda()
   local targetGPU = target:cuda()
   local smtGPU = smt:clone():cuda()
   local tllGPU = tll:clone():cuda()

   -- float
   smt:zeroGradParameters()
   local output = smt:forward{input,target}
   local err = tll:forward(output,target)
   local gradOutput = tll:backward(output,target)
   smt:backward({input,target},gradOutput)
   
   local gradWeightCPU = smt.gradWeight:clone()
   local gradInputCPU = smt.gradInput[1]:clone()

   -- cuda
   smtGPU:zeroGradParameters()
   local outputGPU = smtGPU:forward{inputGPU,targetGPU}
   local errGPU = tllGPU:forward(outputGPU,targetGPU)
   local gradOutputGPU = tllGPU:backward(outputGPU,targetGPU)
   smtGPU:backward({inputGPU,targetGPU},gradOutputGPU) -- note that gradOutputGPU is not contiguous
   
   local gradWeightGPU = smtGPU.gradWeight:clone():float()
   local gradInputGPU = smtGPU.gradInput[1]:clone():float()
   
   mytester:assert(math.abs(err - errGPU) < 0.000001, "SMT err error")
   mytester:assertTensorEq(gradOutput, gradOutputGPU:float(), 0.0000001, "SMT gradOutput error")
   mytester:assertTensorEq(gradWeightCPU, gradWeightGPU, 0.0000001, "SMT gradWeight error")
   mytester:assertTensorEq(gradInputCPU, gradInputGPU, 0.0000001, "SMT gradInput error")
end

function cunnxtest.BlockSparse()
   local nInputBlock = 128
   local nOutputBlock = 128
   local inputSize = 3
   local outputSize = 6
   local inputWindowSize = 7
   local outputWindowSize = 2
   local batchSize = 5
   local lr = 0.1
   
   local input = torch.randn(batchSize,inputWindowSize,inputSize):cuda()
   local gradOutput = torch.randn(batchSize,outputWindowSize,outputSize):cuda()
   local inputIndice = torch.CudaTensor(batchSize, inputWindowSize)
   local outputIndice = torch.CudaTensor(batchSize, outputWindowSize)
   for i=1,batchSize do
      inputIndice[i]:copy(torch.randperm(nInputBlock):narrow(1,1,inputWindowSize))
      outputIndice[i]:copy(torch.randperm(nOutputBlock):narrow(1,1,outputWindowSize))
   end
   local inputScale = torch.randn(batchSize, inputWindowSize):cuda()
   --inputScale:fill(1)
   local outputScale = torch.randn(batchSize, outputWindowSize):cuda()
   --outputScale:fill(1)   
   local gradOutputTable = {gradOutput, {outputIndice, outputScale}}
   local inputTable = {{input, {inputIndice, inputScale}}, {outputIndice, outputScale}}
   local bs = nn.BlockSparse(nInputBlock, inputSize, nOutputBlock, outputSize)
   bs:cuda()
   
   local outputTable = bs:forward(inputTable)
   local output = outputTable[1]
   local gradInputTable = bs:backward(inputTable, gradOutputTable)
   local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
   
   mytester:assertTableEq(output:size():totable(), {batchSize, outputWindowSize, outputSize})
   mytester:assertTableEq(gradInput:size():totable(), {batchSize, inputWindowSize, inputSize})
      
   -- compare
   local cutoff = math.sqrt(inputSize*outputSize)
   bs.batchedGemmMax = cutoff + 10
   
   local message = {'batched', 'streamed'}
   
   for i=1,2 do
      bs:zeroGradParameters()
      
      local outputTable = bs:forward(inputTable)
      local output = outputTable[1]
      local gradInputTable = bs:backward(inputTable, gradOutputTable)
      local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
      
      local exampleIdx = 3
      local input2 = input[exampleIdx]:float()
      local inputIndice2 = inputIndice[exampleIdx]:float():int()
      local outputIndice2 = outputIndice[exampleIdx]:float():int()
      local inputScale2 = inputScale[exampleIdx]:float()
      local outputScale2 = outputScale[exampleIdx]:float()
      local output2 = torch.FloatTensor(outputWindowSize, outputSize):zero()
      local gradOutput2 = gradOutput[exampleIdx]:float()
      local gradInput2 = torch.FloatTensor(inputWindowSize, inputSize):zero()
      local gradOutputScale2 = torch.FloatTensor(outputWindowSize):zero()
      local weight2 = bs.weight:float()
      local bias2 = bs.bias:float()
      
      for j=1,outputWindowSize do
         local output_j = output2[j]
         local outputIdx = outputIndice2[j]
         local outputScale = outputScale2[j]
         local bias = bias2[outputIdx]
         
         output_j:copy(bias)
         
         for i=1,inputWindowSize do
            local input_i = input2[i]
            local inputIdx = inputIndice2[i]
            local inputScale = inputScale2[i]
            local weight = weight2[outputIdx][inputIdx]
             
            output_j:addmv(1, weight, input_i)
         end
         
         output_j:mul(outputScale)
      end   
      
      mytester:assertTensorEq(output[exampleIdx]:float(), output2, precision_forward, 'error on state (forward sparse)'..message[i])
      
      for i=1,inputWindowSize do
         local gradInput_i = gradInput2[i]
         local inputIdx = inputIndice2[i]
         
         for j=1,outputWindowSize do
            local gradOutput_j = gradOutput2[j]
            local outputIdx = outputIndice2[j]
            local outputScale = outputScale2[j]
            local weight = weight2[outputIdx][inputIdx]
      
            gradInput_i:addmv(outputScale, weight:t(), gradOutput_j)
         end
      end 
      
      mytester:assertTensorEq(gradInput[exampleIdx]:float(), gradInput2, precision_backward*10, 'error on state (backward sparse gradInput)'..message[i])
     
      for j=1,outputWindowSize do
         local gradOutput_j = gradOutput2[j]
         local outputScale = outputScale2[j]
         local output_j = output2[j]
         gradOutputScale2[j] = torch.cmul(gradOutput_j, output_j):div(outputScale):sum()
      end
      
      mytester:assertTensorEq(gradOutputScale[exampleIdx]:float(), gradOutputScale2, precision_backward, 'error on state (backward sparse gradOutputScale)'..message[i])

      bs.batchedGemmMax = cutoff - 10
   end
end
   
function cunnxtest.BlockSparse_benchmark()
   local nInputBlock = 384
   local nOutputBlock = 384
   local inputSize = 32
   local outputSize = 32
   local inputWindowSize = 8
   local outputWindowSize = 8
   local batchSize = 128
   local lr = 0.1
   
   local tm, tm2 = {}, {}
   times['BlockSparse vs full dense'] = tm
   times['BlockSparse vs partial dense'] = tm2
   
   local input = torch.randn(batchSize,inputWindowSize,inputSize):cuda()
   local gradOutput = torch.randn(batchSize,outputWindowSize,outputSize):cuda()
   local inputIndice = torch.CudaTensor(batchSize, inputWindowSize)
   local outputIndice = torch.CudaTensor(batchSize, outputWindowSize)
   for i=1,batchSize do
      inputIndice[i]:copy(torch.randperm(nInputBlock):narrow(1,1,inputWindowSize))
      outputIndice[i]:copy(torch.randperm(nOutputBlock):narrow(1,1,outputWindowSize))
   end
   local inputScale = torch.CudaTensor(batchSize, inputWindowSize)
   inputScale:fill(1)
   local outputScale = torch.CudaTensor(batchSize, outputWindowSize)
   outputScale:fill(1)   
   local gradOutputTable = {gradOutput, {outputIndice, outputScale}}
   
   local inputTable = {{input, {inputIndice, inputScale}}, {outputIndice, outputScale}}
   local bs = nn.BlockSparse(nInputBlock, inputSize, nOutputBlock, outputSize)
   bs:cuda()
   bs:zeroGradParameters()
   bs:forward(inputTable)
   bs:backward(inputTable, gradOutputTable, 0.1)
   
   cutorch.synchronize()
   local a = torch.Timer()
   for i=1,nloop do
      local outputTable = bs:forward(inputTable)
      local output = outputTable[1]
      bs:backwardUpdate(inputTable, gradOutputTable, 0.1)
      --bs:updateGradInput(inputTable, gradOutputTable)
      --bs:accGradParameters(inputTable, gradOutputTable)
   end
   cutorch.synchronize()
   
   tm.gpu = a:time().real
   tm2.gpu = a:time().real
   print("BlockSparse time :", tm.gpu)
   bs = nil
   collectgarbage()
   
   local mlp = nn.Linear(nInputBlock*inputSize, nOutputBlock*outputSize)
   mlp:cuda()
   local input3 = torch.randn(batchSize, nInputBlock*inputSize):cuda()
   local gradOutput3 = torch.randn(batchSize, nOutputBlock*outputSize):cuda()
   mlp:forward(input3)
   a:reset()
   for i=1,nloop do
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      mlp:backwardUpdate(input3, gradOutput3, 0.1)
   end
   cutorch.synchronize()
   tm.cpu = a:time().real
   
   mlp = nn.Linear(inputWindowSize*inputSize, outputWindowSize*outputSize)
   mlp:cuda()
   input3 = torch.randn(batchSize, inputWindowSize*inputSize):cuda()
   gradOutput3 = torch.randn(batchSize, outputWindowSize*outputSize):cuda()
   mlp:forward(input3)
   a:reset()
   for i=1,nloop do
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      mlp:backwardUpdate(input3, gradOutput3, 0.1)
   end
   cutorch.synchronize()
   tm2.cpu = a:time().real
end
   
function cunnxtest.BlockSparse_dense()
   -- compare to dense (nn.Linear)
   local nInputBlock = 3
   local nOutputBlock = 2
   local inputSize = 64
   local outputSize = 64
   local inputWindowSize = nInputBlock
   local outputWindowSize = nOutputBlock
   local batchSize = 512
   local lr = 0.1
   
   local input = torch.randn(batchSize,inputWindowSize,inputSize):cuda()
   local gradOutput = torch.randn(batchSize,outputWindowSize,outputSize):cuda()
   local inputIndice = torch.CudaTensor(batchSize, inputWindowSize)
   local outputIndice = torch.CudaTensor(batchSize, outputWindowSize)
   for i=1,batchSize do
      inputIndice[i]:copy(torch.range(1,nInputBlock))
      outputIndice[i]:copy(torch.range(1,nOutputBlock))
   end
   local inputScale = torch.CudaTensor(batchSize, inputWindowSize)
   inputScale:fill(1)
   local outputScale = torch.CudaTensor(batchSize, outputWindowSize)
   outputScale:fill(1)
   local gradOutputTable = {gradOutput, {outputIndice, outputScale}}
   
   local inputTable = {{input, {inputIndice, inputScale}}, {outputIndice, outputScale}}
   local bs = nn.BlockSparse(nInputBlock, inputSize, nOutputBlock, outputSize)
   bs:cuda()
   bs:zeroGradParameters()
   
   local outputTable = bs:forward(inputTable)
   local output = outputTable[1]
   local gradInputTable = bs:backward(inputTable, gradOutputTable)
   local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
   
   local mlp = nn.Linear(nOutputBlock*outputSize, nInputBlock*inputSize)
   mlp.weight = bs.weight:transpose(2, 3):float():resize(nOutputBlock*outputSize, nInputBlock*inputSize)
   mlp.bias = bs.bias:float():resize(nOutputBlock*outputSize)
   mlp.gradWeight = bs.gradWeight:transpose(2, 3):float():resize(nOutputBlock*outputSize, nInputBlock*inputSize)
   mlp.gradBias = bs.gradBias:float():resize(nOutputBlock*outputSize)
   local input2 = input:float():resize(batchSize, inputWindowSize*inputSize)
   local gradOutput2 = gradOutput:float():resize(batchSize, outputWindowSize*outputSize)
   mlp:zeroGradParameters()
   
   local output2 = mlp:forward(input2)
   local gradInput2 = mlp:backward(input2, gradOutput2)
   
   mytester:assertTensorEq(bs.weight:transpose(2, 3):float():resize(nOutputBlock*outputSize, nInputBlock*inputSize), mlp.weight, precision_backward*10, 'error on state (weight dense) ')
   mytester:assertTensorEq(bs.bias:float():resize(nOutputBlock*outputSize), mlp.bias, precision_backward*10, 'error on state (bias dense) ')
   mytester:assertTensorEq(bs.gradWeight:transpose(2, 3):float():resize(nOutputBlock*outputSize, nInputBlock*inputSize), mlp.gradWeight, precision_backward*100, 'error on state (gradWeight dense) ')
   mytester:assertTensorEq(bs.gradBias:float():resize(nOutputBlock*outputSize), mlp.gradBias, precision_backward*100, 'error on state (gradBias dense) ')
   
   bs.maxNorm = 100000
   bs:updateParameters(lr, true)
   mlp:updateParameters(lr)
   
   mytester:assertTensorEq(output:float():resize(batchSize, outputWindowSize*outputSize), output2, precision_forward*10, 'error on state (forward dense) ')
   mytester:assertTensorEq(gradInput:float():resize(batchSize, inputWindowSize*inputSize), gradInput2, precision_backward*10, 'error on state (backward dense) ')
   mytester:assertTensorEq(bs.weight:transpose(2, 3):float():resize(nOutputBlock*outputSize, nInputBlock*inputSize), mlp.weight, precision_backward*10, 'error on state (update weight dense) ')
   mytester:assertTensorEq(bs.bias:float():resize(nOutputBlock*outputSize), mlp.bias, precision_backward*10, 'error on state (update bias dense) ')
end

function cunnxtest.BlockMixture()
   local inputSize = 256
   local nBlock = {128, 256, 128}
   local hiddenSize = {64, 32, 64}
   local gaterSize = 256
   local windowSize = {4, 8, 4}
   local outputSize = 256
   local batchSize = 128
   
   -- experts
   local experts = {}
   local para = nn.ParallelTable()
   para:add(nn.Tanh())
   para:add(nn.Identity())
   
   local expert = nn.Sequential()
   expert:add(nn.BlockSparse(1, inputSize, nBlock[1], hiddenSize[1], true))
   expert:add(para)
   table.insert(experts, expert)
   
   expert = nn.Sequential()
   expert:add(nn.BlockSparse(nBlock[1], hiddenSize[1], nBlock[2], hiddenSize[2], true))
   expert:add(para:clone())
   table.insert(experts, expert)
   
   expert = nn.Sequential()
   expert:add(nn.BlockSparse(nBlock[2], hiddenSize[2], nBlock[3], hiddenSize[3], true))
   expert:add(para:clone())
   table.insert(experts, expert)
   
   expert = nn.Sequential()
   expert:add(nn.BlockSparse(nBlock[3], hiddenSize[3], 1, outputSize, true))
   expert:add(nn.Tanh())
   table.insert(experts, expert)
  
   -- gaters
   local gater = nn.Sequential()
   gater:add(nn.Linear(inputSize, gaterSize))
   gater:add(nn.Tanh())
   local concat = nn.ConcatTable()
   local subGater1 = nn.Sequential()
   subGater1:add(nn.Linear(gaterSize, nBlock[1]))
   subGater1:add(nn.NoisyReLU(windowSize[1]/nBlock[1]))
   subGater1:add(nn.LazyKBest(windowSize[1]))
   concat:add(subGater1)
   
   local subGater2 = nn.Sequential()
   subGater2:add(nn.Linear(gaterSize, nBlock[2]))
   subGater2:add(nn.NoisyReLU(windowSize[2]/nBlock[2]))
   subGater2:add(nn.LazyKBest(windowSize[2]))
   concat:add(subGater2)
   
   local subGater3 = nn.Sequential()
   subGater3:add(nn.Linear(gaterSize, nBlock[3]))
   subGater3:add(nn.NoisyReLU(windowSize[3]/nBlock[3]))
   subGater3:add(nn.LazyKBest(windowSize[3]))
   concat:add(subGater3)
   
   gater:add(concat)
   
   -- mixture
   local bm = nn.BlockMixture(experts, gater)
   bm:cuda()
   
   local input = torch.randn(batchSize, inputSize):cuda()
   local gradOutput = torch.randn(batchSize, outputSize):cuda()
   
   local output = bm:forward(input)
   local gradInput = bm:backwardUpdate(input, gradOutput, 0.1)
   
   mytester:assertTableEq(output:size():totable(), {batchSize, outputSize}, 0.000001)
   mytester:assertTableEq(gradInput:size():totable(), {batchSize, inputSize}, 0.000001)
   
   local tm, tm2, tm3 = {}, {}, {}
   times['BlockMixture vs full dense'] = tm
   times['BlockMixture vs partial dense'] = tm2
   times['gater vs BlockMixture'] = tm3
   
   cutorch.synchronize()
   local a = torch.Timer()
   for i=1,nloop do
      local outputTable = bm:forward(input)
      bm:backwardUpdate(input, gradOutput, 0.1)
      --bm:updateGradInput(input, gradOutput)
      --bm:accGradParameters(input, gradOutput)
   end
   cutorch.synchronize()
   
   tm.gpu = a:time().real
   tm2.gpu = a:time().real
   tm3.cpu = a:time().real
   print("BlockMixture time :", tm.gpu)
   bs = nil
   collectgarbage()
   
   a:reset()
   for i=1,nloop do
      gater:forward(input)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      gater:backwardUpdate(input, bm.gaterGradOutputs, 0.1)
   end
   cutorch.synchronize()
   tm3.gpu = a:time().real
   print("Gater time :", tm3.gpu)
   
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputSize, nBlock[1]*hiddenSize[1]))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(nBlock[1]*hiddenSize[1], nBlock[2]*hiddenSize[2]))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(nBlock[2]*hiddenSize[2], nBlock[3]*hiddenSize[3]))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(nBlock[3]*hiddenSize[3], outputSize))
   mlp:add(nn.Tanh())
   mlp:cuda()
   local input3 = torch.randn(batchSize, inputSize):cuda()
   local gradOutput3 = torch.randn(batchSize, outputSize):cuda()
   mlp:forward(input3)
   a:reset()
   for i=1,nloop do
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      mlp:backwardUpdate(input3, gradOutput3, 0.1)
   end
   cutorch.synchronize()
   tm.cpu = a:time().real
   
   mlp = nn.Sequential()
   mlp:add(nn.Linear(inputSize, windowSize[1]*hiddenSize[1]))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(windowSize[1]*hiddenSize[1], windowSize[2]*hiddenSize[2]))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(windowSize[2]*hiddenSize[2], windowSize[3]*hiddenSize[3]))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(windowSize[3]*hiddenSize[3], outputSize))
   mlp:add(nn.Tanh())
   mlp:cuda()
   input3 = torch.randn(batchSize, inputSize):cuda()
   gradOutput3 = torch.randn(batchSize, outputSize):cuda()
   mlp:forward(input3)
   a:reset()
   for i=1,nloop do
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      mlp:backwardUpdate(input3, gradOutput3, 0.1)
   end
   cutorch.synchronize()
   tm2.cpu = a:time().real
end

function cunnxtest.WindowSparse()
   local inputSize = 32
   local outputSize = 32
   local inputWindowSize = 8
   local outputWindowSize = 8
   local batchSize = 5
   local lr = 0.1
   
   -- windowSparse
   local input = torch.randn(batchSize,inputWindowSize):cuda()
   local gradOutput = torch.randn(batchSize,outputWindowSize):cuda()
   local inputIndice = torch.randperm(inputSize-inputWindowSize):narrow(1,1,batchSize):long()
   local outputIndice = torch.randperm(outputSize-outputWindowSize):narrow(1,1,batchSize):long()
   
   local inputScale = torch.CudaTensor(batchSize, inputWindowSize)
   inputScale:fill(1)
   local outputScale = torch.CudaTensor(batchSize, outputWindowSize)
   outputScale:fill(1)   
   local gradOutputTable = {gradOutput, outputIndice}
   
   local inputTable = {input, inputIndice, outputIndice}
   
   local ws = nn.WindowSparse(inputSize, outputSize, outputWindowSize)
   ws:cuda()
   local cutoff = math.sqrt(inputWindowSize*outputWindowSize)
   ws.batchedGemmMax = cutoff + 10
   
   local message = {'batched', 'streamed'}
   
   -- linear
   local input2 = torch.zeros(batchSize, inputSize):cuda()
   local gradOutput2 = torch.zeros(batchSize, outputSize):cuda()
      
   for i=1,batchSize do
      local inputIdx = inputIndice[i]
      input2[i]:narrow(1, inputIdx, inputWindowSize):copy(input[i])
      local outputIdx = outputIndice[i]
      gradOutput2[i]:narrow(1, outputIdx, outputWindowSize):copy(gradOutput[i])
   end
   
   local mlp = nn.Linear(inputSize, outputSize)
   mlp:cuda()
   mlp.weight = ws.weight:clone()
   mlp.bias = ws.bias:clone()
   
   -- compare
   
   for i=1,2 do
      mlp:zeroGradParameters()
      ws:zeroGradParameters()
      
      local outputTable = ws:forward(inputTable)
      local output = outputTable[1]
      local gradInputTable = ws:backward(inputTable, gradOutputTable)
      local gradInput = gradInputTable[1]
      
      local output2 = mlp:forward(input2)
      local gradInput2 = mlp:backward(input2, gradOutput2)
      
      local output3 = torch.zeros(batchSize, outputWindowSize)
      local gradInput3 = torch.zeros(batchSize, inputWindowSize)
      
      for i=1,batchSize do
         local outputIdx = outputIndice[i]
         output3[i]:copy(output2[i]:narrow(1, outputIdx, outputWindowSize))
         local inputIdx = inputIndice[i]
         gradInput3[i]:copy(gradInput2[i]:narrow(1, inputIdx, inputWindowSize))
      end
      
      mytester:assertTensorEq(output3:float(), output:float(), 0.0001, 'error on state (forward)'..message[i])
      mytester:assertTensorEq(gradInput3:float(), gradInput:float(), 0.0001, 'error on state (backward gradInput)'..message[i])
      ws.batchedGemmMax = cutoff - 10
      
      mytester:assertTensorEq(ws.gradWeight:float(), mlp.gradWeight:float(), 0.0001, 'error on state (backward gradWeight)'..message[i])
      mytester:assertTensorEq(ws.gradBias:float(), mlp.gradBias:float(), 0.0001, 'error on state (backward gradBias)'..message[i])
   end
end

function cunnxtest.WindowSparse_benchmark()
   local inputSize = 10000
   local outputSize = 10000
   local inputWindowSize = 512
   local outputWindowSize = 512
   local batchSize = 256
   --speedup is (forward only)
   -- window/batch streams + gemv     vs gemmBatched
   -- 10k/512/128: 21.9607, 0.1708    vs 10.5454 0.0837
   -- 10k/512/256: 21.98, 0.105       vs 10.4632 0.05
   -- 10k/256/256: 40.68, 0.0944      vs 35.5428 0.0814
   -- 10k/64/128:  38.508 0.0688      vs 131.315 0.2323
   -- 10k/128/128: 39.206 0.0692      vs 85.25   0.15
   local lr = 0.1
   
   local input = torch.randn(batchSize,inputWindowSize):cuda()
   local gradOutput = torch.randn(batchSize,outputWindowSize):cuda()
   local inputIndice = torch.randperm(inputSize-inputWindowSize):narrow(1,1,batchSize):long()
   local outputIndice = torch.randperm(outputSize-outputWindowSize):narrow(1,1,batchSize):long()
   
   local inputScale = torch.CudaTensor(batchSize, inputWindowSize)
   inputScale:fill(1)
   local outputScale = torch.CudaTensor(batchSize, outputWindowSize)
   outputScale:fill(1)   
   local gradOutputTable = {gradOutput, {outputIndice, outputScale}}
   
   local inputTable = {input, inputIndice, outputIndice}
   
   local ws = nn.WindowSparse(inputSize, outputSize, outputWindowSize)
   ws:cuda()
   ws.batchedGemmMax = 200
   
   ws:forward(inputTable)
   local tm, tm2 = {}, {}
   times['WindowSparse vs full dense'] = tm
   times['WindowSparse vs partial dense'] = tm2
   
   cutorch.synchronize()
   local a = torch.Timer()
   for i=1,nloop do
      --experts
      --ws:zeroGradParameters()
      local outputTable = ws:forward(inputTable)
      local output = outputTable[1]
      local gradInputTable = ws:backwardUpdate(inputTable, gradOutputTable, lr)  
      local gradInput = gradInputTable[1]
      --ws:updateGradInput(inputTable, gradOutputTable)
      --ws:accGradParameters(inputTable, gradOutputTable, lr)
      --local gradInputTable = ws:backward(inputTable, gradOutputTable)  
      --local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
      --ws:updateParameters(lr)
   end
   cutorch.synchronize()
   
   tm.gpu = a:time().real
   tm2.gpu = a:time().real
   print("WindowSparse time :", tm.gpu)
   ws = nil
   collectgarbage()
   
   local mlp = nn.Linear(inputSize, outputSize)
   mlp:cuda()
   local input3 = torch.randn(batchSize, inputSize):cuda()
   local gradOutput3 = torch.randn(batchSize, outputSize):cuda()
   mlp:forward(input3)
   a:reset()
   for i=1,nloop do
      --mlp:zeroGradParameters()
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      --mlp:backward(input3, gradOutput3)
      mlp:backwardUpdate(input3, gradOutput3, lr)
      --mlp:updateParameters(lr)
      --mlp.weight:renorm(2, 1, 1)
   end
   cutorch.synchronize()
   tm.cpu = a:time().real
   
   mlp = nn.Linear(inputWindowSize, outputWindowSize)
   mlp:cuda()
   input3 = torch.randn(batchSize, inputWindowSize):cuda()
   gradOutput3 = torch.randn(batchSize, outputWindowSize):cuda()
   mlp:forward(input3)
   a:reset()
   for i=1,nloop do
      --mlp:zeroGradParameters()
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      --mlp:backward(input3, gradOutput3)
      mlp:backwardUpdate(input3, gradOutput3, lr)
      --mlp:updateParameters(lr)
      --mlp.weight:renorm(2, 1, 1)
   end
   cutorch.synchronize()
   tm2.cpu = a:time().real
end

function cunnxtest.Sort()
   local batchSize = 8
   local nInput = 5
   local dim = 2
   local s = nn.Sort(dim)
   local input = torch.randn(batchSize, nInput)
   local output = s:forward(input)
   local gradInput = s:backward(input, output)
   mytester:assertTensorEq(gradInput, input, precision_forward, 'error on state (forward/backward float)')
   s:cuda()
   input = torch.randn(batchSize, nInput):cuda()
   output = s:forward(input)
   gradInput = s:backward(input, output)
   mytester:assertTensorEq(gradInput:float(), input:float(), precision_forward, 'error on state (forward/backward cuda)')
end

function cunnxtest.LazyKBest()
   local batchSize = 8
   local nInput = 16
   local nOutput = 4
   local s = nn.LazyKBest(4)
   s:cuda()
   local input = torch.randn(batchSize, nInput):cuda()
   local output = s:forward(input)
   local gradInput = s:backward(input, output)
   local gradInput2 = input:float():zero()
   local indice = s._indice:type('torch.LongTensor')
   for i=1,input:size(1) do
      gradInput2:select(1, i):indexCopy(1, indice:select(1, i), output[2]:select(1, i):float())
   end

   mytester:assertTensorEq(gradInput:float(), gradInput2:float(), precision_forward, 'error on state (forward/backward)')
end

function cunnxtest.WindowGate()
   local outputWindowSize = 5
   local outputSize = 120
   local inputSize = 7 
   local inputStdv = 2
   local outputStdv = outputWindowSize/2
   local batchSize = 3
   local lr = 0.1
   
   local input = torch.randn(batchSize, inputSize):cuda()
   --[[input = torch.zeros(batchSize, inputSize):cuda()
   input[1][20] = 100
   input[2][1] = 100
   input[3][10] = 100
   input[3][11] = 100--]]
   local gradOutput = torch.randn(batchSize, outputWindowSize):cuda()
   local wg = nn.WindowGate(outputWindowSize, outputSize, inputStdv, outputStdv, lr, 0)
   local sm = nn.SoftMax()
   sm:cuda()
   input = sm:forward(input)
   wg:cuda()
   
   local output = wg:forward(input)
   local gradInput = wg:backward(input, {output[2], gradOutput})  
   
   local input2 = input:clone():float()
   local range = torch.repeatTensor(torch.range(1,input:size(2)):typeAs(input2),input:size(1),1)
   local centroid = torch.cmul(input2, range):sum(2):select(2,1)
   centroid:div(input:size(2))
   centroid:mul(outputSize)
   
   outputIndice = torch.add(centroid, -outputWindowSize/2)
   for i=1,batchSize do
      outputIndice[i] = math.ceil(outputIndice[i])
   end
   outputIndice = outputIndice:long()
   centroid:add(-outputIndice:float()):add(1)
   
   mytester:assertTensorEq(output[1], outputIndice, 0.0000001)
   mytester:assertTensorEq(wg.centroid:float(), centroid, 0.0001)
   
   local output2 = output[2]:float():zero()
   for i=1,batchSize do
      output2[i]:copy(blur(centroid[i], outputStdv, outputWindowSize))
   end
   
   mytester:assertTensorEq(output[2]:float(), output2, 0.00001)
   
   local gradOutput2 = gradOutput:float()
   range = torch.repeatTensor(torch.range(1,outputWindowSize):float(),input:size(1),1)
   range:add(centroid:clone():mul(-1):resize(batchSize, 1):expandAs(range))
   gradOutput2:cmul(output2):cmul(range)
   local gradCentroid = gradOutput2:sum(2)
   gradCentroid:mul(1/(outputStdv*outputStdv))
   gradCentroid:mul(-lr):add(centroid)
   gradCentroid = gradCentroid:add(outputIndice:float()):add(-1):select(2,1)
   gradCentroid:div(outputSize):mul(inputSize)
   
   local target = input2:clone()
   for i=1,batchSize do
      target[i]:copy(blur(gradCentroid[i], inputStdv, inputSize))
   end
   
   local cr = nn.DistNLLCriterion{inputIsProbability=true,targetIsProbability=true}
   local err = cr:forward(input2, target)
   local gradInput2 = cr:backward(input2, target)
   
   mytester:assert(math.abs(err - wg.error:sum()) < 0.00001)
   mytester:assertTensorEq(gradInput2, gradInput:float(), 0.0001)
   
   if true then return end
   cutorch.synchronize()
   local a = torch.Timer()
   for i=1,nloop do
      local output = wg:forward(input)
      local gradInput = wg:backward(input, {gradOutput, output[2]})  
   end
   cutorch.synchronize()
   print("WindowGate time :", a:time().real)
   
end

function cunnxtest.MultinomialStatistics()
   local inputSize = 7 
   local batchSize = 3
   
   local input = torch.randn(batchSize, inputSize)
   local gradOutput = torch.randn(batchSize, inputSize)
   
   local bl = nn.MultinomialStatistics(10)
   local output = bl:forward(input)
   local gradInput = bl:backward(input, gradOutput)
   
   mytester:assertTensorEq(input, output, 0.000001)
   mytester:assertTensorEq(gradOutput, gradInput, 0.000001)
end

function nn.testcudax(tests)
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   math.randomseed(os.time())
   jac = nn.Jacobian
   mytester = torch.Tester()
   mytester:add(cunnxtest)
   mytester:run(tests)
   print ''
   for module,tm in pairs(times) do
      print(module .. ': \t average speedup is ' .. (tm.cpu / (tm.gpu or 1e6)))
   end
   torch.setdefaulttensortype(oldtype)
end


