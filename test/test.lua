require 'torch'
require 'cunn'
require 'nnx'
require 'cunnx'

local cunnxtest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1000
local times = {}
local cunntestx = {}

torch.setdefaulttensortype('torch.FloatTensor')

function cunnxtest.SoftMaxTree()
   local input = torch.randn(5,100)
   local target = torch.IntTensor{20,23,27,10,8}
   local grad = torch.randn(5)
   local root_id = 29
   local hierarchy={
      [29]=torch.IntTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, 
      [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
      [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
      [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
      [8]=torch.IntTensor{24,25,26,27,28}
   }
   local smt = nn.SoftMaxTree(100, hierarchy, root_id)

   local tmF = {}
   local title = string.format('SoftMaxTree forward ')
   times[title] = tmF
   
   local tmB = {}
   local title = string.format('SoftMaxTree backward ')
   times[title] = tmB

   local groundtruthF = smt:forward{input, target}
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruthF = smt:forward{input, target}
   end
   tmF.cpu = a:time().real
   
   local groundtruthB = smt:backward({input, target}, grad)
   a:reset()
   for i = 1,nloop do
      groundtruthB = smt:backward({input, target}, grad)
   end
   tmB.cpu = a:time().real

   input = input:cuda()
   target = target:float():cuda()
   smt:cuda()
   local rescudaF = smt:forward{input, target}
   a:reset()
   for i = 1,nloop do
      rescudaF = smt:forward{input, target}
   end
   cutorch.synchronize()
   tmF.gpu = a:time().real
   
   local rescudaB = smt:backward({input, target}, grad)
   a:reset()
   for i = 1,nloop do
      rescudaB = smt:backward({input, target}, grad)
   end
   cutorch.synchronize()
   tmB.gpu = a:time().real
   
   local error = rescudaF:float() - groundtruthF
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   
   error = rescudaB:float() - groundtruthB
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
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

