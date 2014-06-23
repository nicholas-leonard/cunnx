require 'torch'
require 'cunn'
require 'nnx'
require 'cunnx'

local cunnxtest = {}
local precision_forward = 1e-6
local precision_backward = 1e-6
local nloop = 100
local times = {}
local cunntestx = {}

torch.setdefaulttensortype('torch.FloatTensor')

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
   local groundtruthB = smt:backward({input, target}, grad):clone()
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
   local rescudaB = smt2:backward({input, target}, grad):clone()
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
end

function cunnxtest.BlockSparse()
   local nInputBlock = 128
   local nOutputBlock = 128
   local inputSize = 64
   local outputSize = 64
   local inputWindowSize = 8
   local outputWindowSize = 8
   local batchSize = 512
   local lr = 0.1
   
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
   
   local outputTable = bs:forward(inputTable)
   local output = outputTable[1]
   local gradInputTable = bs:backward(inputTable, gradOutputTable)
   local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
   
   mytester:assertTableEq(output:size():totable(), {batchSize, outputWindowSize, outputSize})
   mytester:assertTableEq(gradInput:size():totable(), {batchSize, inputWindowSize, inputSize})
   
   -- compare for one example
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
   
   for i=1,inputWindowSize do
      local input_i = input2[i]
      local inputScale = inputScale2[i]
      input_i:mul(inputScale)
   end
      
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
   
   mytester:assertTensorEq(output[exampleIdx]:float(), output2, precision_forward, 'error on state (forward sparse)')
   
   for i=1,inputWindowSize do
      local gradInput_i = gradInput2[i]
      local inputIdx = inputIndice2[i]
      local inputScale = inputScale2[i]
      
      for j=1,outputWindowSize do
         local gradOutput_j = gradOutput2[j]
         local outputIdx = outputIndice2[j]
         local outputScale = outputScale2[j]
         local weight = weight2[outputIdx][inputIdx]
   
         gradInput_i:addmv(1, weight:t(), gradOutput_j)
      end
   end 
   
   mytester:assertTensorEq(gradInput[exampleIdx]:float(), gradInput2, precision_backward*10, 'error on state (backward sparse gradInput)')
   
   for j=1,outputWindowSize do
      local gradOutput_j = gradOutput2[j]
      local output_j = output2[j]
      gradOutputScale2[j] = torch.cmul(gradOutput_j, output_j):sum()
   end
   
   mytester:assertTensorEq(gradOutputScale[exampleIdx]:float(), gradOutputScale2, precision_backward, 'error on state (backward sparse gradOutputScale)')
   
   local updates = {}
   for k=1,inputIndice:size(1) do
      for i=1,inputIndice:size(2) do
         local inputIdx = inputIndice[k][i]
         local inputScale = inputScale[k][i]
         
         if inputScale <= 0 then
            break
         end
         
         local update = updates[inputIdx]
         if not update then
            update = {}
            updates[inputIdx] = update
         end
         
         for j=1,outputIndice:size(2) do
            local outputIdx = outputIndice[k][j]
            local outputScale = outputScale[k][j]
            
            if outputScale <= 0 then
               break
            end
            
            local count = update[outputIdx] or 0
            count = count + 1
            update[outputIdx] = count
            
         end
         
      end
   end
   
   for inputIdx, bsUpdate in pairs(bs.updates) do
      local update = updates[inputIdx] or {}
      mytester:assertTableEq(update, bsUpdate, 0, 'error on updates table')
   end
   
   for inputIdx, update in pairs(updates) do
      local bsUpdate = bs.updates[inputIdx] or {}
      mytester:assertTableEq(update, bsUpdate, 0, 'error on updates table')
   end
end
   
function cunnxtest.BlockSparse_benchmark()
   local nInputBlock = 300
   local nOutputBlock = 300
   local inputSize = 32
   local outputSize = 32
   local inputWindowSize = 8
   local outputWindowSize = 8
   local batchSize = 256
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
   
   local gater = nn.BlockSparse(nInputBlock, inputSize, 1, nOutputBlock)
   gater:cuda()
   gradOutputGater = torch.randn(batchSize, nOutputBlock):cuda()
   
   cutorch.synchronize()
   local a = torch.Timer()
   for i=1,nloop do
      --gater
      gater:zeroGradParameters(true)
      gater:forward(inputTable)
      gater:backward(inputTable, gradOutputGater)
      gater:updateParameters(lr, true)
      --experts
      bs:zeroGradParameters(true)
      local outputTable = bs:forward(inputTable)
      local output = outputTable[1]
      --bs:updateGradInput(inputTable, gradOutputTable)
      --bs:accGradParameters(inputTable, gradOutputTable)
      local gradInputTable = bs:backward(inputTable, gradOutputTable)  
      local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
      bs:updateParameters(lr, true) -- also zeros grad parameters
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
      mlp:zeroGradParameters()
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      mlp:backward(input3, gradOutput3)
      mlp:updateParameters(lr)
      mlp.weight:renorm(2, 1, 1)
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
      mlp:zeroGradParameters()
      mlp:forward(input3)
      --mlp:updateGradInput(input3, gradOutput3)
      --mlp:accGradParameters(input3, gradOutput3)
      mlp:backward(input3, gradOutput3)
      mlp:updateParameters(lr)
      mlp.weight:renorm(2, 1, 1)
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
   local gradOutputTable = {gradOutput, {outputIndice, outputScale}}
   
   local inputTable = {{input, {inputIndice, inputScale}}, {outputIndice, outputScale}}
   
   local ws = nn.WindowSparse(inputSize, outputSize)
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
      local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
      
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
   local inputWindowSize = 256
   local outputWindowSize = 256
   local batchSize = 512
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
   
   local inputTable = {{input, {inputIndice, inputScale}}, {outputIndice, outputScale}}
   
   local ws = nn.WindowSparse(inputSize, outputSize)
   ws:cuda()
   ws.batchedGemmMax = 200
   
   ws:forward(inputTable)
   local tm, tm2 = {}, {}
   times['BlockSparse vs full dense'] = tm
   times['BlockSparse vs partial dense'] = tm2
   
   cutorch.synchronize()
   local a = torch.Timer()
   for i=1,nloop do
      --experts
      --ws:zeroGradParameters()
      local outputTable = ws:forward(inputTable)
      local output = outputTable[1]
      local gradInputTable = ws:backwardUpdate(inputTable, gradOutputTable, lr)  
      local gradInput, gradOutputScale = gradInputTable[1][1], gradInputTable[2][2]
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
   s:double()
   input = torch.randn(batchSize, nInput):double()
   output = s:forward(input)
   gradInput = s:backward(input, output)
   mytester:assertTensorEq(gradInput, input, precision_forward, 'error on state (forward/backward double)')
end

function cunnxtest.WindowGate()
   -- outputWindowSize/outputSize == inputWindowSize/inputSize
   local outputWindowSize = 5
   local outputSize = 120
   local inputSize = 20 
   local inputStdv = 2
   local outputStdv = outputWindowSize/2
   local batchSize = 3
   local lr = 0.1
   
   local input = torch.randn(batchSize, inputSize):cuda()
   input = torch.zeros(batchSize, inputSize):cuda()
   input[1][20] = 100000
   input[2][1] = 100000
   input[3][10] = 100000
   input[3][11] = 100000
   local wg = nn.WindowGate(outputWindowSize, outputSize, inputStdv, outputStdv, lr)
   local mlp = nn.Sequential()
   mlp:add(nn.SoftMax())
   mlp:add(wg)
   mlp:cuda()
   
   local output = mlp:forward(input)
   
   print("")
   print("input")
   print(input)
   print("output")
   print(output[1],output[2])
   
end

--cutorch.setDevice(2)

function nn.testcudax(tests)
   math.randomseed(os.time())
   jac = nn.Jacobian
   mytester = torch.Tester()
   mytester:add(cunnxtest)
   mytester:run(tests)
   print ''
   for module,tm in pairs(times) do
      print(module .. ': \t average speedup is ' .. (tm.cpu / (tm.gpu or 1e6)))
   end
end

nn.testcudax({'WindowGate'}) 

