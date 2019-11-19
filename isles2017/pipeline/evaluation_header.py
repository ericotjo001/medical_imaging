from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
from utils.evalobj import EvalObj
import utils.loss as uloss
import utils.metric as me
import models.networks as net
import models.networks_utils as nut

def generic_evaluation_overfit_0001_aux(model,for_evaluation, config_data, ev, ev2, ev3, eval_artifact=True,artifact_epoch=None):
	dice_loss = uloss.SoftDiceLoss()
	
	dice_list = []
	dice_list2 = [] # Adding another dice for double checking
	dice_list3 = []
	if artifact_epoch is None: this_epoch = 'latest'
	else: this_epoch = artifact_epoch
	ev.dice_scores_labelled[this_epoch] = {}
	ev2.dice_scores_labelled[this_epoch] = {}
	ev3.dice_scores_labelled[this_epoch] = {}
	for case_number in for_evaluation:
		x, labels = for_evaluation[case_number] 
		
		model.eval()
		if DEBUG_EVAL_LOOP: model.forward_debug(x); return
		# 1.
		outputs = torch.argmax(model(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		if DEBUG_SHAPES_DURING_EVAL: print("      **DEBUG_SHAPES_DURING_EVAL:\n      labels.shape:%s\n      outputs.shape:%s [before interp]"%(\
			str(labels.shape),str( outputs.shape)))
		# labels.shape:torch.Size([256, 256, 24])
      	# outputs.shape:torch.Size([48, 48, 19]) [before interp]
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		if not check_labels(outputs_label): raise Exception("Output labels not 1.0 or 0.0")
		if DEBUG_EVAL_LOOP: DEBUG_generic_evaluation_overfit_0001(x, labels, outputs, outputs_label); break
		
		d = dice_loss(outputs_label,labels , factor=1)
		dice_score = 1 - d.item()
		dice_list.append(dice_score)
		ev.dice_scores_labelled[this_epoch][case_number] = dice_score

		# 2.
		# Adding another dice for double checking
		outputs_label_reshape = outputs_label.reshape((1,)+outputs_label.shape)
		label_reshape = labels.reshape((1,)+labels.shape)
		dice_score2 = me.DSC(outputs_label_reshape,label_reshape).item()
		dice_list2.append(dice_score2)
		ev2.dice_scores_labelled[this_epoch][case_number] = dice_score2

		# 3.
		model.train()
		outputs = torch.argmax(model(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		d3 = dice_loss(outputs_label,labels , factor=1)
		dice_score3 = 1 - d3.item()
		dice_list3.append(dice_score3)
		ev3.dice_scores_labelled[this_epoch][case_number] = dice_score3

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

def generic_data_loading(config_data, case_numbers_manual=None, case_type='training', verbose=0):
	'''
	- Assume OT is included
	- Assume PWI is not included
	'''
	print("  evaluation_header.py.generic_data_loading()")
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	modalities_dict = {}
	for i, mod in enumerate(config_data['data_modalities']):
		if mod != 'OT': modalities_dict[i] = mod
	no_of_input_channels = len(modalities_dict)
	
	if case_type=='training': 
		case_numbers = range(1,49)
		if DEBUG_EVAL_TRAINING_CASE_NUMBERS is not None: case_numbers = DEBUG_EVAL_TRAINING_CASE_NUMBERS
	elif case_type == 'test':
		case_numbers = range(1,41)
		if DEBUG_EVAL_TEST_CASE_NUMBERS is not None: case_numbers = DEBUG_EVAL_TEST_CASE_NUMBERS

	if case_numbers_manual is not None:	case_numbers = case_numbers_manual

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	resize_shape = tuple(config_data['dataloader']['resize'])
	if DEBUG_dataISLES2017_RESIZE_SHAPE is not None: resize_shape = DEBUG_dataISLES2017_RESIZE_SHAPE
	
	for_evaluation = {}
	case_processed = []
	for case_number in case_numbers:
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
		if one_case is None: continue
		
		if case_type == 'training': 
			s = one_case['imgobj']['OT'].shape
			labels = one_case['imgobj']['OT']
		elif case_type == 'test':
			for xkey in one_case['imgobj']: s = one_case['imgobj'][xkey].shape; break

		x1 = np.zeros(shape=(no_of_input_channels,)+resize_shape)
		if verbose>=50: print("    case_number:%4s | original shape:%s"%(str(case_number),str(s)))
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
		if case_type == 'training':
			labels = torch.tensor(labels).to(torch.int64).to(device=this_device)
			for_evaluation[case_number] = [x,labels]
		elif case_type == 'test':
			for_evaluation[case_number] = [x]
		case_processed.append(case_number)
	print("  data loaded for evaluation!%s"%(str(case_processed)))
	return for_evaluation

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

def generic_evaluation_overfit_0001_aux_save(config_data,ev,ev2,ev3,verbose=0):
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	fullpath = os.path.join(model_dir, 'dice_scores.ev')
	
	dice_scores_to_save = {}
	for i, evN in enumerate([ev,ev2,ev3]):
		dice_scores_to_save['dice_scores'+str(i)] = evN.dice_scores_labelled
		if verbose >=200: print("="*40 + 'dice version' + str(i+1) + "="*30 )
		for epoch in ev.dice_scores_labelled:
			if verbose >=200:
				print("epoch:%s"%(str(epoch)))
				for xkey in ev.dice_scores_labelled[epoch]:	
					print("  %4s: %s"%(str(xkey), str(ev.dice_scores_labelled[epoch][xkey])))

	output = open(fullpath, 'wb')
	pickle.dump(dice_scores_to_save, output)
	output.close()
