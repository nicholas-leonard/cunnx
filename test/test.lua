require 'torch'
require 'cunn'
require 'nnx'
require 'cunnx'

local cunntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}
local cunntestx = {}

torch.setdefaulttensortype('torch.FloatTensor')

function nn.testcuda(tests)
   math.randomseed(os.time())
   jac = nn.Jacobian
   mytester = torch.Tester()
   mytester:add(cunntest)
   mytester:run(tests)
   print ''
   for module,tm in pairs(times) do
      print(module .. ': \t average speedup is ' .. (tm.cpu / (tm.gpu or 1e6)))
   end
end

nn.testcuda()

