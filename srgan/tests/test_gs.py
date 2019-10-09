import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils.utils import *

epsil = 1e-08
a = torch.tensor(1.)# torch.randn(2,3,4)
b = torch.tensor(1.2) # torch.randn(2,3,4)
cs = torch.nn.CosineSimilarity(dim=0, eps=epsil)


bmax = torch.max(b,epsil*torch.ones(b.shape) )
v = a- b * (cs(a,b)/bmax)
print(b)
print(cs(a,b))
print(bmax)

print(v)