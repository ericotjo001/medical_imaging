import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

x = torch.tensor(np.random.randint(0,10,size=(1,1,8))).to(torch.float)
y = torch.tensor(np.random.randint(0,10,size=(1,1,8))).to(torch.float)

print(x)
print(y)

z = torch.cat((x,y),1)
print(z)
print(z.shape)