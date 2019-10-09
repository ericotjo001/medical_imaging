from utils.utils import *
from models.SmallCNN import *

class MultipleConvBlocks(nn.Module):
	"""docstring for MultipleConvBlocks"""
	def __init__(self, multiple_settings=None):
		super(MultipleConvBlocks, self).__init__()
		self.no_of_blocks = 0
		if multiple_settings is not None: self.multiple_convblocks(multiple_settings)

	def multiple_convblocks(self, multiple_settings):
		for i, setting in enumerate(multiple_settings):
			cb_name = str('convblock') + str(i) 
			setattr(self, cb_name, ConvBlocks(setting))
			getattr(self,cb_name).convblock()
			self.no_of_blocks += 1

	def forward(self, x, debug=False):
		if not debug:
			for i in range(self.no_of_blocks):
				cb_name = str('convblock') + str(i) 
				x = getattr(self, cb_name)(x)
		else:
			x = self.forward_debug(x)
		return x

	def forward_debug(self,x):
		for i in range(self.no_of_blocks):
			cb_name = str('convblock') + str(i) 
			x = getattr(self, cb_name).forward_debug(x)
			print("      (%s) x.shape = %s"%(str(cb_name),str(x.shape)))
		return x	

class ConvBlocks(nn.Module):
	def __init__(self, convblock_setting):
		super(ConvBlocks, self).__init__()
		self.convblock_setting = convblock_setting

	def convblock(self, convblock_setting='auto'):
		"""
		One convblock consists of
		1. Conv2D
		2. BatchNorm2D (optional)
		3. Activation (optional)
		"""
		if convblock_setting == 'auto': convblock_setting = self.convblock_setting
		elif convblock_setting is None:
			convblock_setting = {
				'conv':{
					'in_channels': 3,
					'out_channels':6,
					'kernel_size': 3,
					'stride':1,
					'padding':0,
					'dilation':1,
					'groups':1,
					'bias':True,
					'padding_mode':'zeros',
					'conv_name': 'convXXXXXX'
				},
				'bn':{
					'use_batchNorm':True,
					# 'num_features': 6, # same as convblock_setting['conv']['out_channels']
					'eps':1e-05,
					'momentum': 0.1,
					'affine':True,
					'track_running_stats':True,
					'bn_name':'bnXXXXXX'
				},
				'act':{
					'use_act': True,
					'activation_type':'LeakyReLU',
					'activation_name':'actXXXXXX',
					'negative_slope':0.01,
					'inplace':False
				}
				
			}

		############################# Convolution #################################################
		in_channels = convblock_setting['conv']['in_channels']
		out_channels = convblock_setting['conv']['out_channels']
		kernel_size = convblock_setting['conv']['kernel_size']
		stride = convblock_setting['conv']['stride']
		padding = convblock_setting['conv']['padding']
		dilation = convblock_setting['conv']['dilation']
		groups = convblock_setting['conv']['groups']
		bias = convblock_setting['conv']['bias']
		padding_mode = convblock_setting['conv']['padding_mode']
		# conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
		# 	stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
			stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
		setattr(self, convblock_setting['conv']['conv_name'], conv)
		
		############################ batch normalization ############################################
		if convblock_setting['bn']['use_batchNorm']: 
			num_features = out_channels # convblock_setting['num_features']
			eps = convblock_setting['bn']['eps']
			momentum = convblock_setting['bn']['momentum']
			affine = convblock_setting['bn']['affine']
			track_running_stats = convblock_setting['bn']['track_running_stats']
			# bn = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
			setattr(self, convblock_setting['bn']['bn_name'], bn)
		
		################################# Activation ###############################################
		# nn.LeakyReLU(negative_slope=0.01, inplace=False)
		if convblock_setting['act']['use_act']:
			if convblock_setting['act']['activation_type'] == 'LeakyReLU': 
				negative_slope = convblock_setting['act']['negative_slope']
				inplace = convblock_setting['act']['inplace']
				act = nn.LeakyReLU(negative_slope=negative_slope,inplace=inplace)
			elif convblock_setting['act']['activation_type'] == 'Sigmoid': 
				act = nn.Sigmoid()
			setattr(self, convblock_setting['act']['activation_name'], act)
	
	def forward(self, x, debug=False):
		if not debug:
			x = getattr(self, self.convblock_setting['conv']['conv_name'])(x)
			if self.convblock_setting['act']['use_act']: x = getattr(self, self.convblock_setting['act']['activation_name'])(x)
			if self.convblock_setting['bn']['use_batchNorm']: x = getattr(self, self.convblock_setting['bn']['bn_name'])(x)
		else:
			x = self.forward_debug(x)
		return x

	def forward_debug(self,x,squeeze_conv=None):
		x = getattr(self, self.convblock_setting['conv']['conv_name'])(x)
		print("      (%s) x.shape=%s"%(str( self.convblock_setting['conv']['conv_name']),str(x.shape)))
		if self.convblock_setting['act']['use_act']: 
			x = getattr(self, self.convblock_setting['act']['activation_name'])(x)
			print("      (%s) x.shape=%s"%(str(self.convblock_setting['act']['activation_name']),str(x.shape)))
		if self.convblock_setting['bn']['use_batchNorm']: 
			x = getattr(self, self.convblock_setting['bn']['bn_name'])(x)
			print("      (%s) x.shape=%s"%(str(self.convblock_setting['bn']['bn_name']),str(x.shape)))
		return x


class ModulePlus(nn.Module):
	"""docstring for ModulePlus"""
	def __init__(self):
		super(ModulePlus, self).__init__()
		self.latest_epoch = 0
		self.training_cycle = 0
		self.saved_epochs = []

	def perform_routine(self, this_net, config_data):
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 

		if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
		this_net.write_diary(config_data)
		this_net.training_cycle = this_net.training_cycle + 1
		return this_net
		
	def perform_routine_end(self,this_net, config_data, no_of_data_processed=None):
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data,no_of_data_processed=no_of_data_processed )

		this_net.save_models(this_net, config_data)
		this_net.clear_up_models(this_net,config_data, keep_at_most_n_latest_models=config_data['basic']['keep_at_most_n_latest_models'])
		return this_net

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

	def write_diary_post_epoch(self, config_data, no_of_data_processed=None):
		diary_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		diary_full_path = os.path.join(diary_dir,'diary.txt')
		diary_mode = 'a' 
		txt = open(diary_full_path,diary_mode)
		txt.write("  epoch %s time taken: %s [s] %s [min]\n"%(str(self.latest_epoch),str(self.elapsed), str(self.elapsed/60.)))
		if no_of_data_processed is not None: 
			txt.write("    >> no. of data processed in this epoch: %s\n"%(str(no_of_data_processed)))
		txt.close()
		print("  epoch %s time taken: %s [s] %s [min]"%(str(self.latest_epoch),str(round(self.elapsed,2)), str(round(self.elapsed/60.,3))))
		if no_of_data_processed is not None:
			print("    >> no. of data processed in this epoch: %s"%(str(no_of_data_processed)))
			
	def start_timer(self):
		self.start = time.time()
	
	def stop_timer(self):
		self.end = time.time()
		self.elapsed = self.end -self.start

	def save_models(self, model, config_data):
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
		artifact_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(self.latest_epoch) + '.model')

		checker = self.latest_epoch % config_data['basic']['save_model_every_N_epoch']	
		if checker == 0:
			# print("    save_models(). epoch_to_save=%s"%(str(self.latest_epoch)))
			self.saved_epochs.append(self.latest_epoch)
			torch.save(self.state_dict(), artifact_fullpath)
			# output = open(artifact_fullpath, 'wb')
			# pickle.dump(model, output)
			# output.close()
		
		# torch.save(self.state_dict(), main_model_fullpath)
		if os.path.exists(main_model_fullpath): os.remove(main_model_fullpath)
		output2 = open(main_model_fullpath, 'wb')
		pickle.dump(model, output2)
		output2.close()

	def clear_up_models(self,model,config_data, keep_at_most_n_latest_models=None):
		if keep_at_most_n_latest_models is None: return
		# print("clear_up_models()")
		count = 0
		this_epoch = model.latest_epoch
		delete_the_rest = False
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		
		while this_epoch >=0:
			artifact_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(this_epoch) + '.model')
			this_epoch -= 1
			if os.path.exists(artifact_fullpath):
				if not delete_the_rest:
					count+=1
					# print("keep %s"%(artifact_fullpath))
					if count == keep_at_most_n_latest_models: delete_the_rest = True
				else:
					os.remove(artifact_fullpath)
					# print("Cleaning :%s"%(artifact_fullpath))

	def load_state(self, config_data):
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
		pkl_file = open(main_model_fullpath, 'rb')
		model = pickle.load(pkl_file)
		pkl_file.close() 
		return model

	def _init_weight(self):
		# print("Init weight from ModulePlus!")
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				torch.nn.init.kaiming_normal_(m.weight)



def count_parameters(model, print_param=False):
	if print_param:
		for param in model.parameters(): print(param)
	num_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
	num_grad = sum(p.numel() for p in model.parameters())
	print("  networks.py. count_parameters(). With grad: %s, with or without: %s"%(num_with_grad, num_grad))
	return num_with_grad, num_grad