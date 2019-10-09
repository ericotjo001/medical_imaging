from models.networks_components import *
import models.settings_generator as sgen

class SmallCNN(ModulePlus):
	def __init__(self, setup_layers=True, use_CUDA_data_parallel=True):
		super(SmallCNN, self).__init__()

		self.use_CUDA_data_parallel = use_CUDA_data_parallel

		if setup_layers: self.setup_layers()

	def setup_layers(self):
		self.conv_only = ConvBlocks(convblock_setting_conv_only); self.conv_only.convblock()
		self.blocks1 = MultipleConvBlocks(multiple_settings=multiple_settings1)
		self.blocks2 = MultipleConvBlocks(multiple_settings=multiple_settings2)
		self.blocks3 = MultipleConvBlocks(multiple_settings=multiple_settings3)
		self.conv_final = ConvBlocks(convblock_setting_conv_final); self.conv_final.convblock()
		self.drop = nn.Dropout2d(p=0.2)

		if self.use_CUDA_data_parallel:
			for x in self.modules():
				x = torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count()))
		
		self._init_weight()

	def forward(self,x):
		x = self.conv_only(x)
		x = self.blocks1(x,)
		x = self.blocks2(x,)
		x = self.drop(x)
		x = self.blocks3(x,)
		x = self.drop(x)
		x = self.conv_final(x,).squeeze(3).squeeze(2)
		return x

	def forward_debug(self,x):
		print('  models/networks.py. SmallCNN.forward_debug()')
		print('    [-1] x.shape:%s'%(str(x.shape)))
		debug = True
		x = self.conv_only(x,debug=debug)
		print('    [0]    x.shape:%s'%(str(x.shape)))
		x = self.blocks1(x,debug=debug)
		print('    [1]    x.shape:%s'%(str(x.shape)))
		x = self.blocks2(x,debug=debug)
		x = self.drop(x)
		print('    [2]    x.shape:%s'%(str(x.shape)))
		x = self.blocks3(x,debug=debug)
		x = self.drop(x)
		print('    [3]    x.shape:%s'%(str(x.shape)))
		x = self.conv_final(x,debug=debug).squeeze(3).squeeze(2)
		print('    [4]    x.shape:%s'%(str(x.shape)))
		return x


		
class Generator0001(ModulePlus):
	"""docstring for Generator0001"""
	def __init__(self, setup_layers=True, use_CUDA_data_parallel=True):
		super(Generator0001, self).__init__()
	
		self.use_CUDA_data_parallel = use_CUDA_data_parallel
		self.blocks1 = MultipleConvBlocks(multiple_settings=sgen.multiple_settings1)
		self.blocks2 = MultipleConvBlocks(multiple_settings=sgen.multiple_settings2)
		self.blocks3 = MultipleConvBlocks(multiple_settings=sgen.multiple_settings3)
		self.conv_final = ConvBlocks(sgen.convblock_setting_conv_final); self.conv_final.convblock()
		if setup_layers: self.setup_layers()

	def setup_layers(self):
		self.conv_only = ConvBlocks(sgen.convblock_setting_conv_only); self.conv_only.convblock()

		if self.use_CUDA_data_parallel:
			for x in self.modules():
				x = torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count()))
		
		self._init_weight()

	def forward(self,x):
		debug = 0
		x = self.conv_only(x,debug=debug)
		x = self.blocks1(x,debug=debug)
		x = self.blocks2(x,debug=debug)
		x = self.blocks3(x,debug=debug)
		x = self.conv_final(x,debug=debug).squeeze(3).squeeze(2)
		return x

	def forward_debug(self,x):
		print('  models/networks.py. SmallCNN.Generator0001()')
		print('    [-1] x.shape:%s'%(str(x.shape)))
		debug = True
		x = self.conv_only(x,debug=debug)
		print('    [0]    x.shape:%s'%(str(x.shape)))
		x = self.blocks1(x,debug=debug)
		print('    [1]    x.shape:%s'%(str(x.shape)))
		x = self.blocks2(x,debug=debug)
		print('    [2]    x.shape:%s'%(str(x.shape)))
		x = self.blocks3(x,debug=debug)
		print('    [3]    x.shape:%s'%(str(x.shape)))
		x = self.conv_final(x,debug=debug).squeeze(3).squeeze(2)
		print('    [4]    x.shape:%s'%(str(x.shape)))
		return x

	def forward2(self,x,pos=0):
		debug = 0
		x = self.conv_only(x,debug=debug)
		if pos==1: return x
		x = self.blocks1(x,debug=debug)
		if pos==2: return x
		x = self.blocks2(x,debug=debug)
		if pos==3: return x
		x = self.blocks3(x,debug=debug)
		if pos==4: return x
		x = self.conv_final(x,debug=debug).squeeze(3).squeeze(2)
		return x