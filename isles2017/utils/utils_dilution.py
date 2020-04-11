
from utils.utils import *

class FilterConv3D():
	def __init__(self, conv_obj=None):
		super(FilterConv3D, self).__init__()
		
		if conv_obj == 'average':
			self.conv_obj = torch.nn.Conv3d(1, 1, 3, stride=1, padding=1, bias=False)
			self.conv_obj.weight.data = self.conv_obj.weight.data*0 + 1./len(self.conv_obj.weight.data.reshape(-1))		
		else:
			if not type(conv_obj) == type(torch.nn.Conv3d(1, 1, 3)):
				raise RuntimeError('Insert a torch.nn.Conv3d object.')
			self.conv_obj = conv_obj

	def channel_wise_conv(self, x):
		s = x.shape
		out = None
		for c in range(s[1]):
			x3d_vol = x[:1,c:c+1,]
			avg = self.conv_obj(x3d_vol)
			if out is None:
				out = avg
			else:
				out = torch.cat((out,avg), 1)

			# print(x3d_vol)
			# print(avg)
			# print(x3d_vol.shape)
		return out