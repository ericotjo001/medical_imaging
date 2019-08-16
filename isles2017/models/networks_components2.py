from utils.utils import *

DEBUG_COMPONENT = 0

class conv3Dbatchnorm(nn.Module):
	"""docstring for conv3Dbatchnorm"""
	def __init__(self,in_channels,out_channels, kernel, stride,padding,bias=True,dilation=1,is_batchnorm=True, device=None):
		super(conv3Dbatchnorm, self).__init__()

		conv = nn.Conv3d(in_channels,out_channels,kernel_size=kernel,padding=padding,stride=stride,bias=bias,dilation=dilation)
		if is_batchnorm: self.convblock = nn.Sequential(conv, nn.BatchNorm3d(out_channels))
		else: self.convblock = nn.Sequential(conv)

	def forward(self, inputs):
		outputs = self.convblock(inputs)
		return outputs



class conv3DBatchNormRelu(nn.Module):
	def __init__(self,in_channels,n_filters,k_size,stride,padding,bias=True,dilation=1,is_batchnorm=True,):
		super(conv3DBatchNormRelu, self).__init__()

		conv_mod = nn.Conv3d(in_channels,n_filters,kernel_size=k_size,padding=padding,stride=stride,bias=bias,dilation=dilation,)
		if is_batchnorm: self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm3d(int(n_filters)), nn.ReLU(inplace=True))
		else: self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

	def forward(self, inputs):
		outputs = self.cbr_unit(inputs)
		return outputs

'''
For segnet
'''
class segnetDown2(nn.Module):
	def __init__(self, in_size, out_size, mp_kernel=2, mp_stride=2):
		super(segnetDown2, self).__init__()
		self.conv1 = conv3DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv3DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool_with_argmax = nn.MaxPool3d(mp_kernel,mp_stride, return_indices=True)

	def forward(self, inputs):
		outputs = self.conv1(inputs)
		outputs = self.conv2(outputs)
		unpooled_shape = outputs.size()
		outputs, indices = self.maxpool_with_argmax(outputs)
		return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
	def __init__(self, in_size, out_size, mp_kernel=2, mp_stride=2):
		super(segnetDown3, self).__init__()
		self.conv1 = conv3DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv3DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.conv3 = conv3DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool_with_argmax = nn.MaxPool3d(mp_kernel,mp_stride, return_indices=True)

	def forward(self, inputs):
		outputs = self.conv1(inputs)
		outputs = self.conv2(outputs)
		outputs = self.conv3(outputs)
		unpooled_shape = outputs.size()
		outputs, indices = self.maxpool_with_argmax(outputs)
		return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
	def __init__(self, in_size, out_size, upool_kernel=2, upool_stride=2):
		super(segnetUp2, self).__init__()
		self.unpool = nn.MaxUnpool3d(upool_kernel, upool_stride)
		self.conv1 = conv3DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv3DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, inputs, indices, output_shape):
		outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		return outputs


class segnetUp3(nn.Module):
	def __init__(self, in_size, out_size, upool_kernel=2, upool_stride=2):
		super(segnetUp3, self).__init__()
		self.unpool = nn.MaxUnpool3d(upool_kernel, upool_stride)
		self.conv1 = conv3DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv3DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv3 = conv3DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, inputs, indices, output_shape):
		outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		outputs = self.conv3(outputs)
		return outputs

'''
For PSP
'''

class pyramidPooling(nn.Module):
	def __init__(self, in_channels, pool_sizes, model_name="pspnet", fusion_mode="cat", is_batchnorm=True):
		super(pyramidPooling, self).__init__()
		

		bias = not is_batchnorm
		self.paths = []
		out_channels = int(in_channels / len(pool_sizes))

		for i in range(len(pool_sizes)):
			self.paths.append(conv3DBatchNormRelu(in_channels,out_channels,1,1,0,bias=bias,is_batchnorm=is_batchnorm,))
		self.path_module_list = nn.ModuleList(self.paths)
		self.pool_sizes = pool_sizes
		self.model_name = model_name
		self.fusion_mode = fusion_mode

	def forward(self, x):
		d, h, w = x.shape[2:]
		if self.training or self.model_name != "icnet": 
			k_sizes = []
			strides = []
			for pool_size in self.pool_sizes:
				assert(len(pool_size) == 3)
				ta = int(d / pool_size[0]); 
				tb = int(h / pool_size[1]); 
				tc = int(w / pool_size[2]); 
				if ta == 0 : ta = 1
				if tb == 0 : tb = 1 
				if tc == 0 : tc = 1
				k_sizes.append((ta, tb, tc))
				strides.append((ta, tb, tc))
		else:  # [commmend copied as is] eval mode and icnet: pre-trained for 1025 x 2049
			''' I added the d dimension to the kernel sizes and strides'''
			k_sizes = [(3,8, 15), (3,13, 25), (3,17, 33), (3,33, 65)]
			strides = [(1,5, 10), (1,10, 20), (1,16, 32), (1,33, 65)]

		if self.fusion_mode == "cat":  # pspnet: concat (including x)
			output_slices = [x]

			for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
				out = F.avg_pool3d(x, k_sizes[i], stride=strides[i], padding=0)
				if self.model_name != "icnet":
					out = module(out)
				out = F.interpolate(out, size=(d, h, w), mode="trilinear", align_corners=True)
				output_slices.append(out)

			return torch.cat(output_slices, dim=1)
		else:  # icnet: element-wise sum (including x)
			pp_sum = x

			for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
				out = F.avg_pool3d(x, k_sizes[i], stride=strides[i], padding=0)
				if self.model_name != "icnet":
					out = module(out)
				out = F.interpolate(out, size=(h, w), mode="trilinear", align_corners=True)
				pp_sum = pp_sum + out

		return pp_sum




class residualBlockPSP(nn.Module):
	def __init__(self, n_blocks, in_channels, mid_channels, out_channels,
		stride, dilation=1, include_range="all", is_batchnorm=True,):
		super(residualBlockPSP, self).__init__()

		if dilation > 1: stride = 1

		# residualBlockPSP = convBlockPSP + identityBlockPSPs
		layers = []
		if include_range in ["all", "conv"]:
			layers.append(bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation, is_batchnorm=is_batchnorm))
		if include_range in ["all", "identity"]:
			for i in range(n_blocks - 1):
				layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation, is_batchnorm=is_batchnorm))

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class bottleNeckPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels, stride, dilation=1, is_batchnorm=True):
		super(bottleNeckPSP, self).__init__()
		bias = not is_batchnorm
		self.cbr1 = conv3DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
		if dilation > 1:
			self.cbr2 = conv3DBatchNormRelu(mid_channels, mid_channels,3,stride=stride,padding=dilation,bias=bias,dilation=dilation,is_batchnorm=is_batchnorm,)
		else:
			self.cbr2 = conv3DBatchNormRelu(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm,)
		self.cb3 = conv3Dbatchnorm(mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
		self.cb4 = conv3Dbatchnorm(in_channels,out_channels,1,stride=stride,padding=0,bias=bias,is_batchnorm=is_batchnorm,)

	def forward(self, x):
		conv = self.cb3(self.cbr2(self.cbr1(x)))
		residual = self.cb4(x)
		return F.relu(conv + residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, stride, dilation=1, is_batchnorm=True):
		super(bottleNeckIdentifyPSP, self).__init__()
		bias = not is_batchnorm

		self.cbr1 = conv3DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)

		if dilation > 1:
			self.cbr2 = conv3DBatchNormRelu(mid_channels,mid_channels,3,stride=1,padding=dilation,bias=bias,dilation=dilation,is_batchnorm=is_batchnorm,)
		else:
			self.cbr2 = conv3DBatchNormRelu(mid_channels,mid_channels,3,stride=1,padding=1,bias=bias,dilation=1,is_batchnorm=is_batchnorm,)
		self.cb3 = conv3Dbatchnorm(mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)

	def forward(self, x):
		residual = x
		x = self.cb3(self.cbr2(self.cbr1(x)))
		return F.relu(x + residual, inplace=True)