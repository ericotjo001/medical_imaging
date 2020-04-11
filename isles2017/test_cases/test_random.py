import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

for i in range(10):
	x = np.random.randint(-10,10)
	print(x)