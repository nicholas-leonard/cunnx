require 'torch'
require 'cunn'
require 'nnx'
require 'cunnx'

cutorch.setDevice(1)

local cunnxtest = {}
local precision_forward = 1e-6
local precision_backward = 1e-6
local nloop = 1000
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

nn.testcudax()

