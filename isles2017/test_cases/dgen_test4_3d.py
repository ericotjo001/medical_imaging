import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DGenSample

def create_3d_from_2d_by_interp(x, shape2d=(192,192), this_shape=(19,192,192)):
	x3d = np.zeros(shape=(1,1,3,)+shape2d)
	x3d[0,0,1,:,:] = x # for torch, need (batch,channel, D,H,W)
	x3d[0,0,0,90:100,90:100] = 0.1+np.random.uniform(0,0.2)
	x3d[0,0,2,90:100,90:100] = 0.1+np.random.uniform(0,0.2)
	x3din = F.interpolate(torch.tensor(x3d),size=this_shape,mode='trilinear',align_corners=True)
	x3din = x3din.squeeze().detach().numpy()
	return x3din

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

x3din = create_3d_from_2d_by_interp(x,shape2d=x.shape, this_shape=(19,192,192))
x3din_un = create_3d_from_2d_by_interp(x1,shape2d=x.shape, this_shape=(19,192,192))
y3din = create_3d_from_2d_by_interp(y,shape2d=x.shape, this_shape=(19,192,192))
y3din = (y3din>0.8).astype(float)
print(np.any(np.isnan(y3din)))

LIST_of_indices_to_observe = [[0,1,2],[6,7,8], [10,11,12],[16,17,18]]

for indices_to_observe in LIST_of_indices_to_observe:
	fig = plt.figure()
	ax = fig.add_subplot(331)
	im1 =ax.imshow(x3din[indices_to_observe[0]])
	ax.set_title('healthy [%s]'%(str(indices_to_observe[0])))
	plt.colorbar(im1)

	ax2 = fig.add_subplot(332)
	im2 =ax2.imshow(x3din[indices_to_observe[1]])
	ax2.set_title('healthy [%s]'%(str(indices_to_observe[1])))
	plt.colorbar(im2)

	ax3 = fig.add_subplot(333)
	im3 = ax3.imshow(x3din[indices_to_observe[2]])
	ax3.set_title('healthy [%s]'%(str(indices_to_observe[2])))
	plt.colorbar(im3)

	ax4 = fig.add_subplot(334)
	im4 =ax4.imshow(x3din_un[indices_to_observe[0]])
	ax4.set_title('unhealthy')
	plt.colorbar(im4)

	ax5 = fig.add_subplot(335)
	im5 =ax5.imshow(x3din_un[indices_to_observe[1]])
	ax5.set_title('unhealthy')
	plt.colorbar(im5)

	ax6 = fig.add_subplot(336)
	im6 = ax6.imshow(x3din_un[indices_to_observe[2]])
	ax6.set_title('unhealthy')
	plt.colorbar(im6)

	ax7 = fig.add_subplot(337)
	im7 =ax7.imshow(y3din[indices_to_observe[0]])
	ax7.set_title('gt')
	plt.colorbar(im7)

	ax8 = fig.add_subplot(338)
	im8 =ax8.imshow(y3din[indices_to_observe[1]])
	ax8.set_title('gt')
	plt.colorbar(im8)

	ax9 = fig.add_subplot(339)
	im9 = ax9.imshow(y3din[indices_to_observe[2]])
	ax9.set_title('gt')
	plt.colorbar(im9)

	plt.tight_layout()
plt.show()