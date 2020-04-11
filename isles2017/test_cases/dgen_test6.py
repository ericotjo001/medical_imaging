import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DG3D
dg = DG3D(unit_size=(192,192), depth=19)

XH, XU, Y =[],[],[]
for i in range(1,10):
	print(i)
	x_healthy, x_unhealthy, y_lesion = dg.generate_data(random_mode=True)
	XH.append(x_healthy)
	XU.append(x_unhealthy)
	Y.append(y_lesion)
print('done generating')
fig = plt.figure()
for i in range(1,10):
	ax = fig.add_subplot(330+i)
	im1 =ax.imshow(XH[i-1][10])
	plt.colorbar(im1)
	plt.tight_layout()
fig2 = plt.figure()
for i in range(1,10):
	ax = fig2.add_subplot(330+i)
	im1 =ax.imshow(XU[i-1][10])
	plt.colorbar(im1)
	plt.tight_layout()
fig3 = plt.figure()
for i in range(1,10):
	ax = fig3.add_subplot(330+i)
	im1 =ax.imshow(Y[i-1][10])
	plt.colorbar(im1)
	plt.tight_layout()
plt.show()