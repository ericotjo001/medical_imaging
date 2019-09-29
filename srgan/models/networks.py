from models.networks_components import *
import models.settings_discriminator as sdisc

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

class Discriminator(ModulePlus):
	def __init__(self, setup_layers=True, use_CUDA_data_parallel=True ):
		super(Discriminator, self).__init__()

		self.use_CUDA_data_parallel = use_CUDA_data_parallel

		if setup_layers: self.setup_layers()

	def setup_layers(self):
		self.conv_only = ConvBlocks(sdisc.convblock_setting_conv_only); self.conv_only.convblock()
		self.blocks1 = MultipleConvBlocks(multiple_settings=sdisc.multiple_settings1)
		self.blocks2 = MultipleConvBlocks(multiple_settings=sdisc.multiple_settings2)
		self.conv_final = ConvBlocks(sdisc.convblock_setting_conv_final); self.conv_final.convblock()
		self.fc =  nn.Linear(sdisc.fc_input, sdisc.fc_output)
		self.fc2 =  nn.Linear(sdisc.fc_output, 2)
		# self.drop2d = nn.Dropout2d(p=0.5)
		# self.drop = nn.Dropout(p=0.2)
		self.sg =  nn.LeakyReLU() #nn.Sigmoid()
		"""
		See that we use 2*sg()-1 in the feed forward part of the network
		
		"""
		
		if self.use_CUDA_data_parallel:
			for x in self.modules():
				x = torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count()))
		
		self._init_weight()

	def forward(self,x):
		debug = False
		x0 = self.conv_only(x[0],debug=debug)
		x0 = self.blocks1(x0,debug=debug)
		# x0 = self.drop2d(x0)
		x0 = self.blocks2(x0,debug=debug)
		# x0 = self.drop2d(x0)
		x0 = self.conv_final(x0,debug=debug).squeeze(3).squeeze(2)
		x = torch.cat((x0,x[1]),dim=1)
		x = self.sg(self.fc(x))
		# x = self.drop(x)
		x = self.sg(self.fc2(x))
		# x = self.drop(x)
		return x

	def forward_debug(self,x):
		print('  models/networks.py. Discriminator.forward_debug()')
		print('    [-1] x[0].shape:%s, x[1].shape:%s'%(str(x[0].shape),str(x[1].shape)))
		debug = True
		x0 = self.conv_only(x[0],debug=debug)
		print('    [0]    x0.shape:%s'%(str(x0.shape)))
		x0 = self.blocks1(x0,debug=debug)
		# x0 = self.drop2d(x0)
		print('    [1]    x0.shape:%s'%(str(x0.shape)))
		x0 = self.blocks2(x0,debug=debug)
		# x0 = self.drop2d(x0)
		print('    [2]    x0.shape:%s'%(str(x0.shape)))
		x0 = self.conv_final(x0,debug=debug).squeeze(3).squeeze(2)
		print('    [3]    x0.shape:%s'%(str(x0.shape)))
		x = torch.cat((x0,x[1]),dim=1)
		print('    [4]    x.shape:%s'%(str(x.shape)))
		x = self.sg(self.fc(x))
		# x = self.drop(x)
		x = self.sg(self.fc2(x))
		# x = self.drop(x)
		print('    [5]    x.shape:%s'%(str(x.shape)))
		return x
		
		