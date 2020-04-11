from utils.utils import *
import models.networks_LRP as lr

class ConvBlocksUNet(nn.Module):
	def __init__(self, label, batch_norm, with_LRP=False, is_the_first_block=False):
		super(ConvBlocksUNet, self).__init__()
		# self.this_device = device
		self.label = label
		self.batch_norm = batch_norm
		self.with_LRP = with_LRP
		self.is_the_first_block = is_the_first_block
		self.relprop_max_min = None # typically [0,1]
	
	def conv_three_blocks(self, in_channels, out_channels, kernel_sizes, paddings, strides, dilations):
		'''
		we label by the "levels" starting from 1. After each maxpool, level increases by 1.
		The original paper https://arxiv.org/abs/1606.06650 has 3 blocks at each level before devonolution
		and have 4 levels. THen it is followed by 3 levels of deconvolution layers with concatenation.
		'''
		for i, param in enumerate(zip(in_channels, out_channels, kernel_sizes, paddings, strides, dilations)):
			if i == 0 and self.is_the_first_block: 
				conv, convbn = self.convblock(param[0], param[1], param[2], padding=param[3], stride=param[4], 
					dilation=param[5], is_the_first_layer=True)
			else: 
				conv, convbn = self.convblock(param[0], param[1], param[2], padding=param[3], stride=param[4], dilation=param[5])	
			setattr(self, 'conv_'+ str(self.label) + "_" +str(i), conv)
			if self.batch_norm: setattr(self, 'bn3d_'+ str(self.label)+ "_"  +str(i), convbn)	
			setattr(self, 'act_'+str(self.label)+"_"+str(i), lr.LeakyReLU_LRP(0.2, inplace=True))
		return

	def convblock(self, in_channel, out_channel, kernel_size, padding=0, stride=1, dilation=1, batch_norm=True, is_the_first_layer=False):
		if not self.with_LRP:
			conv = nn.Conv3d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, dilation=dilation)#.to(device=self.this_device)  
			if self.batch_norm: convbn = nn.BatchNorm3d(out_channel)##.to(device=self.this_device)
			else: convbn = None
		else:
			conv = lr.Conv3dLRP(in_channel, out_channel, kernel_size, padding=padding, stride=stride, dilation=dilation)#.to(device=self.this_device)  
			if self.batch_norm: convbn = lr.BatchNorm3dLRP(out_channel)##.to(device=self.this_device)
			else: convbn = None
			if is_the_first_layer:
				conv.not_first_layer = False
				conv.relprop_max_min = self.relprop_max_min
		return conv, convbn

	def forward(self, x, save_for_relprop=True):
		if DEBUG_networks_components_COMPONENT: print("  Components: ConvBlocksUNet()")
		if self.with_LRP and save_for_relprop : self.X = x
		else: self.X = None
		for i in range(2):	
			x = getattr(self,'conv_'+ str(self.label) + "_"+ str(i))(x, save_for_relprop=save_for_relprop)
			if self.batch_norm: x = getattr(self,'bn3d_'+ str(self.label) + "_" + str(i))(x)
			x = getattr(self, 'act_'+str(self.label)+"_"+str(i))(x)
			# x = F.relu(x)

			if DEBUG_networks_components_COMPONENT: print("    x.shape:%s"%(str(x.shape)))
		return x

	def relprop(self,R, verbose=0):
		if verbose>99: 			
			print("  ConvBlocksUNet. relprop().")
			ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5
			print("    sqr sum=%s np.max(R)=%s, np.min(R)=%s"%(str(ss),str(np.max(R.detach().cpu().numpy())),str(np.min(R.detach().cpu().numpy()))))

		for i in range(2)[::-1]:	
			R = getattr(self, 'act_'+str(self.label)+"_"+str(i)).relprop(R)
			if self.batch_norm: R = getattr(self,'bn3d_'+ str(self.label) + "_" + str(i)).relprop(R)
			R = getattr(self,'conv_'+ str(self.label) + "_"+ str(i)).relprop(R,verbose=verbose)
			if verbose>99: 
				ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5
				print("    sqr sum=%s np.max(R)=%s, np.min(R)=%s"%(str(ss),str(np.max(R.detach().cpu().numpy())),str(np.min(R.detach().cpu().numpy()))))
		return R


class ModulePlus(nn.Module):
	"""docstring for ModulePlus"""
	def __init__(self):
		super(ModulePlus, self).__init__()
		self.latest_epoch = 0
		self.training_cycle = 0
		self.saved_epochs = []
		self.training_lifetime = 0.

	def write_diary(self, config_data):
		diary_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		# diary_full_path = os.path.join(diary_dir,'diary.txt')
		diary_name = 'diary.txt'
		
		header = "\n%s\nTraining cycle [%s]:\n%s\n"%("="*60, str(self.training_cycle), "="*60)
		DictionaryTxtManager(diary_dir,diary_name,dictionary_data=config_data,header=header)


		# if not os.path.exists(diary_dir): os.mkdir(diary_dir)
		# diary_mode = 'a' 
		# if not os.path.exists(diary_full_path): diary_mode = 'w'
		
		# txt = open(diary_full_path,diary_mode)
		# txt.write("\n%s\nTraining cycle [%s]:\n%s\n"%("="*60, str(self.training_cycle), "="*60))
		# for x in config_data:
		# 	if not isinstance(config_data[x],dict) :
		# 		txt.write("  %s : %s [%s]"%(x, config_data[x],type(config_data[x])))
		# 	else:
		# 		txt.write("  %s :"%(x))
		# 		for y in config_data[x]:
		# 			txt.write("    %s : %s [%s]"%(y, config_data[x][y], type(config_data[x][y])))
		# 			txt.write("\n")
		# 	txt.write("\n")
		# txt.close()

	def write_diary_post_epoch(self, config_data, no_of_data_processed=None):
		diary_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		diary_full_path = os.path.join(diary_dir,'diary.txt')
		diary_mode = 'a' 
		txt = open(diary_full_path,diary_mode)
		txt.write("  epoch %s time taken: %s [s] %s [min]."%(str(self.latest_epoch),str(round(self.elapsed,2)), str(round(self.elapsed/60.,3))))
		if no_of_data_processed is not None: 
			txt.write("    [%s data batch(es)]\n"%(str(no_of_data_processed)))
		txt.write("\n    total training lifetime: %s [min] %s [hr]\n"%(str(round(self.training_lifetime/60.,2)),str(round(self.training_lifetime/3660.,2))))
		txt.close()
		print("  epoch %s time taken: %s [s] %s [min]."%(str(self.latest_epoch),str(round(self.elapsed,2)), \
			str(round(self.elapsed/60.,3))))
		print("    total training lifetime: %s [min] %s [hr]"%(str(round(self.training_lifetime/60.,2)),str(round(self.training_lifetime/3660.,2))),end='')
		if no_of_data_processed is not None:
			print(" [%s data batch(es)]"%(str(no_of_data_processed)))


	def start_timer(self):
		self.start = time.time()
	
	def stop_timer(self):
		self.end = time.time()
		self.elapsed = self.end -self.start
		self.training_lifetime = self.training_lifetime + self.elapsed

	def save_models(self, model, config_data):
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
		artifact_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(self.latest_epoch) + '.model')

		checker = self.latest_epoch % config_data['basic']['save_model_every_N_epoch']	
		if checker == 0:
			# print("    save_models(). epoch_to_save=%s"%(str(self.latest_epoch)))
			self.saved_epochs.append(self.latest_epoch)
			torch.save(self.state_dict(), artifact_fullpath)
		
		if os.path.exists(main_model_fullpath): os.remove(main_model_fullpath)
		output2 = open(main_model_fullpath, 'wb')
		pickle.dump(model, output2)
		# pickle.dump(model, output2)
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

	def post_process_sequence(self, this_net, config_data, no_of_data_processed=None):
		this_net.latest_epoch = this_net.latest_epoch + 1
		self.write_diary_post_epoch(config_data,no_of_data_processed=no_of_data_processed )
		self.save_models(this_net, config_data)
		self.clear_up_models(this_net,config_data, keep_at_most_n_latest_models=config_data['basic']['keep_at_most_n_latest_models'])
		return this_net
