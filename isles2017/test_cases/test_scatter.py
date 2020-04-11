import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

this_size = (100)
x = np.random.normal(0,1,size=this_size)
y = np.random.normal(0,1,size=this_size)
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.random.normal(0,1,size=this_size)
y = np.random.normal(0,1,size=this_size)
ax.scatter(x,y, label='ggwp', s=3)
x = np.random.normal(0,1,size=this_size)
y = np.random.normal(0,1,size=this_size)
ax.scatter(x,y, label='ggwp2', s=3)
plt.legend()
plt.show()