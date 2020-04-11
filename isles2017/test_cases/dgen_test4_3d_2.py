import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DG3D
dg = DG3D(unit_size=(192,192), depth=19)
x3din, x3din_un, y3din = dg.generate_data()

LIST_of_indices_to_observe = [[9,10,11]]# [[0,1,2],[6,7,8], [10,11,12],[16,17,18]]

print("**")
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