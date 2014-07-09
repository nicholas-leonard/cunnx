require "cutorch"
require "nn"
require "cunn"
require "nnx"
require "libcunnx"

torch.include('cunnx', 'WindowSparse.lua')
torch.include('cunnx', 'WindowGate.lua')
torch.include('cunnx', 'WindowGate2.lua')
torch.include('cunnx', 'WindowMixture.lua')
torch.include('cunnx', 'Balance.lua')
torch.include('cunnx', 'LinearNoBias.lua')

torch.include('cunnx', 'BlockSparse.lua')
torch.include('cunnx', 'BlockMixture.lua')
torch.include('cunnx', 'Sort.lua')
torch.include('cunnx', 'LazyKBest.lua')
torch.include('cunnx', 'NoisyReLU.lua')
torch.include('cunnx', 'TypeDecorator.lua')
