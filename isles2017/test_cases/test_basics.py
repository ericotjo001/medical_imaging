import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as pl

import sys

sg = nn.Sigmoid()

x = torch.tensor(np.linspace(-100,100,1000),dtype=torch.float)
y = 2 * sg(0.1*x)-1

plt.figure()
plt.scatter(x,y,s=3)
plt.show()