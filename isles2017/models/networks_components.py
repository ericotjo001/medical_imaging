from utils.utils import *

DEBUG_COMPONENT = 0

class ConvBlocksUNet(nn.Module):
	"""docstring for ConvBlocksUNet"""
	def __init__(self, label, batch_norm, device=None):
		super(ConvBlocksUNet, self).__init__()
		self.this_device = device
		self.label = label
		self.batch_norm = batch_norm
	
	def conv_three_blocks(self, in_channels, out_channels, kernel_sizes, paddings, strides, dilations):
		'''
		we label by the "levels" starting from 1. After each maxpool, level increases by 1.
		The original paper https://arxiv.org/abs/1606.06650 has 3 blocks at each level before devonolution
		and have 4 levels. THen it is followed by 3 levels of devonolution layers with concatenation.
		'''
		for i, param in enumerate(zip(in_channels, out_channels, kernel_sizes, paddings, strides, dilations)):
			conv, convbn = self.convblock(param[0], param[1], param[2], padding=param[3], stride=param[4], dilation=param[5])
			setattr(self, 'conv_'+ str(self.label) + "_" +str(i), conv)
			if self.batch_norm: setattr(self, 'bn3d_'+ str(self.label)+ "_"  +str(i), convbn)	
		return

	def convblock(self, in_channel, out_channel, kernel_size, padding=0, stride=1, dilation=1, batch_norm=True):
		conv = nn.Conv3d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, dilation=dilation).to(device=self.this_device)  
		if self.batch_norm: convbn = nn.BatchNorm3d(out_channel).to(device=self.this_device)
		else: convbn = None
		return conv, convbn

	def forward(self, x):
		if DEBUG_COMPONENT: print("  Components: ConvBlocksUNet()")
		for i in range(2):	
			x = getattr(self,'conv_'+ str(self.label) + "_"+ str(i))(x)
			x = F.relu(x)
			if self.batch_norm: x = getattr(self,'bn3d_'+ str(self.label) + "_" + str(i))(x)

			if DEBUG_COMPONENT: print("    x.shape:%s"%(str(x.shape)))
		return x

class ConvBlocks(nn.Module):
	""" ConvBlocks"""
	def __init__(self, device=None):
		super(ConvBlocks, self).__init__()
		self.this_device=device
		self.number_of_blocks = 0
		self.without_bn = True
		
	def convblocks(self,in_channels, out_channels, kernel_sizes, paddings, strides, dilations):
		for i, param in enumerate(zip(in_channels, out_channels, kernel_sizes, paddings, strides, dilations)):
			conv, convbn = self.convblock(param[0], param[1], param[2], padding=param[3], stride=param[4], dilation=param[5])
			setattr(self, 'conv'+str(i), conv)
			setattr(self, 'bn3d'+str(i), convbn)
			self.number_of_blocks = self.number_of_blocks + 1
		return 

	def convblock(self, in_channel, out_channel, kernel_size, padding=0, stride=1, dilation=1):
		conv = nn.Conv3d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, dilation=dilation)
		convbn = nn.BatchNorm3d(out_channel)
		if self.this_device is not None:
			conv = conv.to(device=self.this_device)  
			convbn = convbn.to(device=self.this_device)
		return conv, convbn

	def convblocks_without_bn(self,in_channels, out_channels, kernel_sizes, paddings, strides, dilations):
		for i, param in enumerate(zip(in_channels, out_channels, kernel_sizes, paddings, strides, dilations)):
			conv = self.convblock_without_bn(param[0], param[1], param[2], padding=param[3], stride=param[4], dilation=param[5])
			setattr(self, 'conv'+str(i), conv)
			self.number_of_blocks = self.number_of_blocks + 1
		return 

	def convblock_without_bn(self, in_channel, out_channel, kernel_size, padding=0, stride=1, dilation=1):
		conv = nn.Conv3d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, dilation=dilation)
		if self.this_device is not None:
			conv = conv.to(device=self.this_device)  
		return conv

	def forward(self, x):
		for i in range(self.number_of_blocks):
			x = getattr(self,'conv'+str(i))(x)
			x = F.elu(x)
			if not self.without_bn: x = getattr(self,'bn3d'+str(i))(x)
		return x

class ConvBlocksPool(nn.Module):
	""" ConvBlocks"""
	def __init__(self, device=None):
		super(ConvBlocksPool, self).__init__()
		self.this_device=device
		self.number_of_blocks = 0
		
	def convblocks_with_pool(self,in_channels, out_channels, kernel_sizes, paddings, strides, dilations):
		for i, param in enumerate(zip(in_channels, out_channels, kernel_sizes, paddings, strides, dilations)):
			conv, pool = self.convblock_with_pool(param[0], param[1], param[2], padding=param[3], stride=param[4], dilation=param[5])
			setattr(self, 'conv'+str(i), conv)
			setattr(self, 'pool'+str(i), pool)
			self.number_of_blocks = self.number_of_blocks + 1
		return 

	def convblock_with_pool(self, in_channel, out_channel, kernel_size, padding=0, stride=1, dilation=1):
		conv = nn.Conv3d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, dilation=dilation)
		pool =  nn.MaxPool3d((1,2,2), stride=(1,2,2), ceil_mode=True)
		if self.this_device is not None:
			conv = conv.to(device=self.this_device)  
			pool = pool.to(device=self.this_device)
		return conv,  pool

	def forward(self, x):
		for i in range(self.number_of_blocks):
			x = getattr(self,'conv'+str(i))(x)
			x = F.elu(x)
			x = getattr(self,'pool'+str(i))(x)
		return x



class ModulePlus(nn.Module):
	"""docstring for ModulePlus"""
	def __init__(self):
		super(ModulePlus, self).__init__()
		self.latest_epoch = 0
		self.training_cycle = 0
		self.saved_epochs = []

	def write_diary(self, config_data):
		diary_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		diary_full_path = os.path.join(diary_dir,'diary.txt')
		if not os.path.exists(diary_dir): os.mkdir(diary_dir)
		diary_mode = 'a' 
		if not os.path.exists(diary_full_path): diary_mode = 'w'
		
		txt = open(diary_full_path,diary_mode)
		txt.write("\n%s\nTraining cycle [%s]:\n%s\n"%("="*60, str(self.training_cycle), "="*60))
		for x in config_data:
			if not isinstance(config_data[x],dict) :
				txt.write("  %s : %s [%s]"%(x, config_data[x],type(config_data[x])))
			else:
				txt.write("  %s :"%(x))
				for y in config_data[x]:
					txt.write("    %s : %s [%s]"%(y, config_data[x][y], type(config_data[x][y])))
					txt.write("\n")
			txt.write("\n")
		txt.close()

	def write_diary_post_epoch(self, config_data):
		diary_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		diary_full_path = os.path.join(diary_dir,'diary.txt')
		diary_mode = 'a' 
		txt = open(diary_full_path,diary_mode)
		txt.write("  epoch %s time taken: %s [s] %s [min]\n"%(str(self.latest_epoch),str(self.elapsed), str(self.elapsed/60.)))
		txt.close()
		print("  epoch %s time taken: %s [s] %s [min]"%(str(self.latest_epoch),str(self.elapsed), str(self.elapsed/60.)))

	def start_timer(self):
		self.start = time.time()
	
	def stop_timer(self):
		self.end = time.time()
		self.elapsed = self.end -self.start

	def save_models(self, model, config_data):
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
		artifact_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(self.latest_epoch) + '.model')

		checker = self.latest_epoch % config_data['basic_1']['save_model_every_N_epoch']	
		if checker == 0:
			# print("save_models(). epoch_to_save=%s"%(str(self.latest_epoch)))
			self.saved_epochs.append(self.latest_epoch)
			output = open(artifact_fullpath, 'wb')
			pickle.dump(model, output)
			output.close()
			
		output = open(main_model_fullpath, 'wb')
		pickle.dump(model, output)
		output.close()

	def load_state(self, config_data):
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
		pkl_file = open(main_model_fullpath, 'rb')
		model = pickle.load(pkl_file)
		pkl_file.close() 
		return model

	def _init_weight(self):
		print("Init weight from ModulePlus!")
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				torch.nn.init.kaiming_normal_(m.weight)