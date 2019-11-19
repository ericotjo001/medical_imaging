from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
import models.networks as net
import utils.vis as vi
import utils.loss as uloss

import models.networks_utils as nut
from pipeline.evaluation import get_modalities_0001
from pipeline.evaluation import generic_data_loading


##########################################################
# Generic
##########################################################

def save_one_lrp_output(case_number,y,Rc, config_data,case_type='training', output_lrp_folder_name='output_lrp'):
	'''
	y is output predicted by the network.
	Rc is LRP output
	'''
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	lrp_dir = os.path.join(model_dir,output_lrp_folder_name)
	if not os.path.exists(lrp_dir): os.mkdir(lrp_dir)
	lrp_fullpath = os.path.join(lrp_dir,'yR_'+case_type+'_'+str(case_number)+'.lrp')

	output = open(lrp_fullpath, 'wb')
	pickle.dump([y,Rc], output)
	output.close()


class LRP_diary_manager(object):
	def __init__(self, config_data, training_cycle,more_text=None):
		super(LRP_diary_manager, self).__init__()
		self.total_time = 0.
		self.diary_name = 'diary_LRP.txt'
		self.diary_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		self.diary_full_path = os.path.join(self.diary_dir,self.diary_name)

		self.LRP_diary(config_data, training_cycle, more_text=more_text)	

	def LRP_diary(self, config_data, training_cycle, more_text = None):
		header = "\n%s\nTraining cycle [%s]:\n%s\n"%("="*60, str(training_cycle), "="*60)
		if more_text is not None:
			header = header + '\n' + more_text +'\n'
		DictionaryTxtManager(self.diary_dir,self.diary_name,dictionary_data=config_data['LRP'],header=header)
		
	def add_dice_score_per_case_number(self, case_number, dice_score):
		txt = open(self.diary_full_path,'a')
		txt.write("case_number:%s\n  dice=%s\n"%(str(case_number),str(dice_score)))
		txt.close()

	def start_timing_per_case(self):
		self.start = time.time()

	def stop_timing_per_case(self):
		self.end = time.time()
		self.elapsed = self.end - self.start
		self.total_time += self.elapsed
		print("  time taken: %s [s]. total time spent: %s [min]"%(str(self.elapsed),str(self.total_time/60.)))
		txt = open(self.diary_full_path,'a')
		txt.write("  time taken: %s [s] %s [min]\n"%(str(self.elapsed),str(self.elapsed/60.)))
		txt.write("     total time spent: %s [min]\n"%(str(self.total_time/60.)))
		txt.close()		

def convert_prediction_to_correct_size(this_net, x, labels):
	R = this_net(x).contiguous()
	outputs = torch.argmax(this_net(x).contiguous(),dim=1)
	outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
	outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
	y = outputs_label.detach().cpu().numpy()
	return R, y, outputs_label

def compute_dice_and_update_LRP_diary(LRP_diary_manager ,case_number, dice_loss, outputs_label, labels):
	d = dice_loss(outputs_label, labels , factor=1)
	dice_score = 1 - d.item()
	LRP_diary_manager.add_dice_score_per_case_number(case_number, dice_score)

def convert_LRP_output_to_correct_size(R, x, y, sx, s, no_of_modalities, verbose=0):
	# we need to resize channel by channel
	Rc = np.zeros(shape=(no_of_modalities,)+s)
	for c in range(x.shape[1]): 
		Rc_part = centre_crop_tensor(R[c], sx[2:] ).permute(2,1,0)
		Rc_part = interp3d(Rc_part,s).detach().cpu().numpy()
		# print("  np.max(Rc_part) = %s, np.min(Rc_part)=%s"%(str(np.max(Rc_part)),str(np.min(Rc_part))))
		Rc[c] = Rc_part	
	Rmax, Rmin = np.max(Rc), np.min(Rc)
	if verbose>=100:
		print("  Rc.shape = %s [%s]\n  y.shape = %s [%s]"%(str(Rc.shape),str(type(Rc)),str(y.shape),str(type(y))))
		print("  Rmax = %s, Rmin = %s"%(str(Rmax),str(Rmin)))
	return Rc

##########################################################
# lrp_UNet3D_filter_sweeper_0003
##########################################################

def lrp_UNet3D_filter_sweeper_0003_prepare_data(config_data, case_number, modalities_dict, verbose=0):
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	
	case_type = 'training'
	data_modalities = config_data['data_modalities']
	model_label_name = config_data['model_label_name']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],model_label_name)
	output_folder_name = 'lrp_filter_sweep_mode_0002'
	lrp_dir = os.path.join(model_dir,output_folder_name)
	
	from pipeline.visual_header import load_lrp_sweep
	from dataio.dataISLES2017 import ISLES2017mass

	fullpath_to_lrpd_file = os.path.join(lrp_dir,'lrp_dict_' + str(case_type) + "_" + str(case_number)+'.lrpd')
	if verbose>=250: 
		print(fullpath_to_lrpd_file)
		print("    %64s|%s"%('xkey','lrp_output_dictionary[xkey].shape'))
	lrp_output_dictionary = load_lrp_sweep(fullpath_to_lrpd_file)
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
	x1, _ = ISLESDATA.load_type0001_get_raw_input(one_case, modalities_dict)	

	for xkey in lrp_output_dictionary:
		if verbose>=250: print("    %64s|%s"%(str(xkey), str(lrp_output_dictionary[xkey].shape)))
	if verbose>=250: print("      x1.shape:%s"%(str(x1.shape)))

	return x1, lrp_output_dictionary

##########################################################
# lrp_UNet3D_filter_sweeper_0001
##########################################################

def predictions_for_LRP(output_dictionary, y, labels):
	output_dictionary[('OT')] = labels
	output_dictionary[('y')] = y

def raw_mode_LRP(output_dictionary, relprop_config, this_net, R, x, y, sx, s, no_of_modalities, verbose=0):
	this_relprop_config = relprop_config.copy()
	this_relprop_config['normalization'] = 'raw'
	R1 = this_net.relprop(R, this_relprop_config).squeeze()
	Rc = convert_LRP_output_to_correct_size(R1, x, y, sx, s, no_of_modalities, verbose=0)
	if verbose > 99:
		print('  raw. Rc.shape = %s'%(str(Rc.shape)))
	output_dictionary[('raw')] = Rc


def filter_mode_LRP(output_dictionary, relprop_config, this_net, R, x, y, sx, s, no_of_modalities, filter_sweep_options, filter_mode='fraction_pass_filter',verbose=0):
	filter_grid_iter = filter_sweep_options[filter_mode]
	for nfilter, pfilter in filter_grid_iter:
		this_relprop_config = relprop_config.copy()
		this_relprop_config['normalization'] = filter_mode
		this_relprop_config[filter_mode]['positive'] = pfilter
		this_relprop_config[filter_mode]['negative'] = nfilter 
		R1 = this_net.relprop(R, this_relprop_config).squeeze()
		Rc = convert_LRP_output_to_correct_size(R1, x, y, sx, s, no_of_modalities, verbose=0)
		if verbose > 99:
			print('  pass filter <<%10s,%10s>>. Rc.shape = %s'%(str(nfilter),str(pfilter),str(Rc.shape)))
		output_dictionary[(filter_mode, tuple(nfilter), tuple(pfilter))] = Rc

def build_output_dictionary(relprop_config,this_net, R, x, y, labels, sx, s, no_of_modalities, filter_sweep_options, verbose=0):
	output_dictionary = {}
	predictions_for_LRP(output_dictionary, y, labels.detach().cpu().numpy())
	raw_mode_LRP(output_dictionary, relprop_config, this_net, R, x, y, sx, s, no_of_modalities, verbose=0)
	filter_mode_LRP(output_dictionary,relprop_config, this_net, R, x, y, sx, s, no_of_modalities, filter_sweep_options, filter_mode='fraction_pass_filter',verbose=0)
	filter_mode_LRP(output_dictionary,relprop_config, this_net, R, x, y, sx, s, no_of_modalities, filter_sweep_options, filter_mode='fraction_clamp_filter',verbose=0)
	if verbose>=100:
		for xkey in output_dictionary:
			print("  %60s | %-20s"%(str(xkey), str(output_dictionary[xkey].shape)))
	return output_dictionary

def save_one_LRP_output_dictionary(output_dictionary, case_number ,config_data, case_type='training', output_folder_name='lrp_filter_sweep'):
	# print("save_one_LRP_output_dictionary()...")
	dict_name = 'lrp_dict_' + case_type + '_' + str(case_number) + '.lrpd'
	folderpath = os.path.join(config_data['working_dir'],config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	if not os.path.exists(folderpath): raise Exception('folder not found!')
	folderpath = os.path.join(folderpath, output_folder_name)
	if not os.path.exists(folderpath): os.mkdir(folderpath)

	fullpath = os.path.join(folderpath, dict_name)
	output = open(fullpath, 'wb')
	pickle.dump(output_dictionary, output)
	output.close()

	
##########################################################
# 
##########################################################