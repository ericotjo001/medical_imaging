from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
from utils.evalobj import EvalObj
import utils.loss as uloss
import utils.metric as me
import models.networks as net

# DEBUG_TEST_CASE_NUMBERS = [1,6] # range(1,10) # default is None # for training
# DEBUG_TEST_CASE_NUMBERS = [1,2,3] # None # range(1,10) # default is None # for test
# DEBUG_EVAL_LOOP = 0

def generic_evaluation_overfit_0001_aux(model,for_evaluation, config_data, ev, ev2, ev3, eval_artifact=True,artifact_epoch=None):
	dice_loss = uloss.SoftDiceLoss()
	
	dice_list = []
	dice_list2 = [] # Adding another dice for double checking
	dice_list3 = []
	for case_number in for_evaluation:
		x, labels = for_evaluation[case_number] 
		
		model.eval()
		# 1.
		outputs = torch.argmax(model(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		if not check_labels(outputs_label): raise Exception("Output labels not 1.0 or 0.0")
		if DEBUG_EVAL_LOOP: DEBUG_generic_evaluation_overfit_0001(x, labels, outputs, outputs_label); break
		
		d = dice_loss(outputs_label,labels , factor=1)
		dice_score = 1 - d.item()
		dice_list.append(dice_score)

		# 2.
		# Adding another dice for double checking
		outputs_label_reshape = outputs_label.reshape((1,)+outputs_label.shape)
		label_reshape = labels.reshape((1,)+labels.shape)
		dice_score2 = me.DSC(outputs_label_reshape,label_reshape).item()
		dice_list2.append(dice_score2)

		# 3.
		model.train()
		outputs = torch.argmax(model(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		d3 = dice_loss(outputs_label,labels , factor=1)
		dice_score3 = 1 - d3.item()
		dice_list3.append(dice_score3)

	if eval_artifact:
		save_epoch = artifact_epoch
		ev.get_dice_score_at_epoch(save_epoch, dice_list)
		# Adding another dice for double checking
		ev2.get_dice_score_at_epoch(save_epoch, dice_list2)
		ev3.get_dice_score_at_epoch(save_epoch, dice_list3)
	else:
		# evaluating the latest model
		ev.get_dice_score_latest(dice_list)
		ev2.get_dice_score_latest(dice_list2)
		ev3.get_dice_score_latest(dice_list3)

		ev.save_evaluation(config_data,report_name='report_eval_mode.txt')
		ev2.save_evaluation(config_data,report_name='report_eval_mode2.txt')
		ev3.save_evaluation(config_data,report_name='report_train_mode.txt')
	return

def generic_data_loading(config_data):
	'''
	- Assume OT is included
	- Assume PWI is not included
	'''
	print("Calling generic_data_loading()")
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	modalities_dict = {}
	for i, mod in enumerate(config_data['data_modalities']):
		if mod != 'OT': modalities_dict[i] = mod
	no_of_input_channels = len(modalities_dict)
	
	case_type = 'training'
	case_numbers = range(1,49)
	if DEBUG_EVAL_TRAINING_CASE_NUMBERS is not None: case_numbers = DEBUG_EVAL_TRAINING_CASE_NUMBERS

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	resize_shape = tuple(config_data['dataloader']['resize'])

	for_evaluation = {}
	for case_number in case_numbers:
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
		if one_case is None: continue
		s = one_case['imgobj']['OT'].shape
		x1 = np.zeros(shape=(no_of_input_channels,)+resize_shape)
		labels = one_case['imgobj']['OT']
		print("  case_number:%s | original shape:%s"%(str(case_number),str(labels.shape)))
		for modality_key in modalities_dict:
			# print("    modalities_dict[%s]:%s"%(str(modality_key),str(modalities_dict[modality_key])))
			if modalities_dict[modality_key] == 'OT': continue
			# print("    modalities_dict[modality_key]:%s"%(modalities_dict[modality_key]))
			x1_component = one_case['imgobj'][modalities_dict[modality_key]]
			x1_component = normalize_numpy_array(x1_component,
				target_min=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][0],
				target_max=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][1],
				source_min=np.min(x1_component),#config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][0],
				source_max=np.max(x1_component),#config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][1], 
				verbose = 0)
			x1_component = torch.tensor(x1_component)
			x1[modality_key,:,:,:] = interp3d(x1_component,resize_shape)
		
		#x1 is now C,W,H,D
		x1s = x1.shape
		x = torch.tensor([x1]).to(torch.float).to(device=this_device)
		# print("x.shape:%s"%(str(x.shape)))
		x = x.permute(0,1,4,3,2)
		# print("x.shape after permute :%s"%(str(x.shape)))
		labels = torch.tensor(labels).to(torch.int64).to(device=this_device)
		for_evaluation[case_number] = [x,labels]
	print("data loaded for evaluation!")
	return for_evaluation

def load_training_data_for_submission(config_data):
	print("evaluation_header.py. load_training_data_for_submission()")

	return

def DEBUG_generic_evaluation_overfit_0001(x, labels, outputs, outputs_label):
	print("DEBUG_generic_evaluation_overfit_0001()")
	print("  x.shape:%s"%(str(x.shape)))
	print("  labels.shape:%s"%(str(labels.shape)))
	print("  outputs.shape:%s"%(str(outputs.shape)))
	print("  outputs_label.shape:%s"%(str(outputs_label.shape)))

def check_labels(y):
	temp = y.detach().cpu().numpy().reshape(-1)
	unique_list = list(sorted(set(temp)))
	if DEBUG_EVAL_LOOP: print("  unique_list:",unique_list)
	for possible_list in [[0.],[1.],[0.,1.]]:
		if np.all(unique_list==possible_list): return True
	return False
