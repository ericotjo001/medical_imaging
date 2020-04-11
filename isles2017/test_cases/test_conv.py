import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

MODE = 3
x = torch.tensor(np.random.randint(0,10,size=(1,2,4,4,4))).to(torch.float)
x = x*0+1

if MODE == 1:
	# x.shape:torch.Size([1, 6, 19, 48, 48])
	def channel_wise_conv(x, conv_obj):
		s = x.shape
		out = None
		for c in range(s[1]):
			x3d_vol = x[:1,c:c+1,]
			avg = conv_obj(x3d_vol)
			if out is None:
				out = avg
			else:
				out = torch.cat((out,avg), 1)

			# print(x3d_vol)
			# print(avg)
			# print(x3d_vol.shape)
		return out

	avg_filter_3d = torch.nn.Conv3d(1, 1, 7, stride=1, padding=1, bias=False)
	avg_filter_3d.weight.data = avg_filter_3d.weight.data*0 + 1./len(avg_filter_3d.weight.data.reshape(-1))
	# print(avg_filter_3d.weight.data)
	# print(x)
	out = channel_wise_conv(x, avg_filter_3d)
	print(out.shape)
	print(avg_filter_3d.weight.data.shape)
	print(avg_filter_3d.weight.data.reshape(-1))
elif MODE ==2:
	from utils.utils_dilution import FilterConv3D
	con = FilterConv3D(conv_obj='average')
	out = con.channel_wise_conv(x)

	print(out)
elif MODE ==3:
	avg_filter_3d = torch.nn.Conv3d(1, 1, 3, stride=1, padding=1, bias=False)
	avg_filter_3d.weight.data = avg_filter_3d.weight.data*0 + 1.
	avg_filter_3d.weight.data[0,0,1,1,1]=99.
	print(avg_filter_3d.weight.data)
	print(avg_filter_3d.weight.data.shape)




