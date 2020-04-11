import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DGenSample

dg = DGenSample(unit_size=(192,192))
dim = len(dg.unit_size)

x = dg.get_sample()
x[x>0.6] = 1.
y = dg.get_sample_lesion()
coin_flip = np.random.uniform(0,1)
if coin_flip>0.5:
	sgn = -1 
	random_effect_factor = np.random.uniform(0.5,0.8, size=y.shape)
else:
	sgn = 1
	random_effect_factor = np.random.uniform(0.3,0.4, size=y.shape)
x1 = np.clip(x +sgn* x*y*random_effect_factor,0,1)
y= (y-y*(x>0.99).astype(float))
# xs = dg.get_sample_solid_structure()

fig = plt.figure()
ax = fig.add_subplot(221)
im1 =ax.imshow(x)
ax.set_title('healthy')
plt.colorbar(im1)

ax2 = fig.add_subplot(222)
im2 =ax2.imshow(y)
ax2.set_title('lesion')
plt.colorbar(im2)

ax3 = fig.add_subplot(223)
im3 = ax3.imshow(x1)
ax3.set_title('not healthy[%s]'%(str(sgn)))
plt.colorbar(im3)

plt.tight_layout()
plt.show()
