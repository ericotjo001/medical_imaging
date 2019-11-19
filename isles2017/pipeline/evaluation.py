from pipeline.evaluation_header import *


def evaluation_UNet3D_overfit(config_data):
	print("evaluation_UNet3D_overfit()")
	generic_evaluation_overfit_0001(config_data)

def evaluation_UNet3D_overfit_submission(config_data):
	print("evaluation_UNet3D_overfit_submission()")
	evaluation_for_submission_overfit_0001(config_data)

def evaluation_UNet3D_test_submission(config_data):
	print('evaluation_UNet3D_test_submission()')
	evaluation_for_test_submission_0001(config_data)

def get_modalities_0001(config_data):
	modalities_dict = {}
	for i, mod in enumerate(config_data['data_modalities']): 
		if mod != 'OT': modalities_dict[i] = mod
	no_of_input_channel = len(modalities_dict)
	print("  no_of_input_channel:%s"%(str(no_of_input_channel)))
	print("    ",end='')
	for xkey in modalities_dict: print("%5s "%(str(modalities_dict[xkey])),end=' | ')
	print()
	return modalities_dict, no_of_input_channel


'''
More implementation details
'''

def generic_evaluation_overfit_0001(config_data):
	'''
	Assume OT is loaded
	Assume PWI is not included.
	'''
	print("  pipeline/evaluation.py. generic_evaluation_overfit_0001()")
	ev = EvalObj()
	ev2 = EvalObj() # anotehr dice computation
	ev3 = EvalObj() # same as ev, but without eval()

	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
	model_type = config_data['training_mode']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
	this_net.eval()


	for_evaluation = generic_data_loading(config_data)
	for save_epoch in this_net.saved_epochs:
		artifact_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(save_epoch) + '.model')
		if not os.path.exists(artifact_model_fullpath): continue

		if model_type == 'UNet3D': artifact_net = net.UNet3D(no_of_input_channel=no_of_input_channels, with_LRP=True) 
		else: raise Exception("Invalid mode")
		
		artifact_net.load_state_dict(torch.load(artifact_model_fullpath))
		
		print("    Looking at model after epoch %s"%(str(save_epoch)))
		generic_evaluation_overfit_0001_aux(artifact_net,for_evaluation,config_data, ev, ev2, ev3, eval_artifact=True, artifact_epoch=save_epoch)
		if DEBUG_EVAL_LOOP: break
	if DEBUG_EVAL_LOOP: return

	print("    Looking at the latest model.")
	generic_evaluation_overfit_0001_aux(this_net,for_evaluation,config_data, ev, ev2, ev3, eval_artifact=False)
	generic_evaluation_overfit_0001_aux_save(config_data,ev,ev2,ev3,verbose=0)	


def evaluation_for_submission_overfit_0001(config_data):
	'''
	This evaluation is compatible with training using data loaded from
	- ISLES2017mass() load_many_cases_type0003().
	'''
	print("evaluation_for_submission_overfit_0001()")
	ev = EvalObj()
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']

	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	model_type = config_data['training_mode']
	this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
	this_net.eval()

	case_type = 'training'
	for_evaluation = generic_data_loading(config_data, case_type=case_type)
	dice_check1_list, dice_check2_list = [], []
	for case_number in for_evaluation:
		x, labels = for_evaluation[case_number]
		s = labels.shape 
		outputs = torch.argmax(this_net(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		if DEBUG_SHAPES_DURING_EVAL: print("      **DEBUG_SHAPES_DURING_EVAL:\n\t\tlabels.shape:%s\n\t\toutputs.shape:%s [before interp]"%(\
			str(labels.shape),str( outputs.shape)))
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		y_pred = outputs_label.detach().cpu().numpy()
		if  DEBUG_SHAPES_DURING_EVAL: print("\t\ty_pred.shape:%s"%(str(y_pred.shape)))

		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
		ISLESDATA.save_one_case(y_pred.squeeze(), one_case, case_type, case_number,config_data, desc='etjoa001_UNet3D_'+str(case_number))
		dice_check1, dice_check2 = ev.save_one_case_evaluation(case_number, outputs_label.reshape((1,)+s), labels.reshape((1,)+s), config_data, dice=True, filename='output_train_report.txt')
		dice_check1_list.append(dice_check1)
		dice_check2_list.append(dice_check2)
	print("dice_check lists:%s [%s]"%(str(np.mean(dice_check1_list)),str(np.mean(dice_check2_list))))

	
def evaluation_for_test_submission_0001(config_data,verbose=0):
	print('    evaluation_for_test_submission_0001()')

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']

	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	model_type = config_data['training_mode']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
	this_net.eval()

	case_type = 'test'
	for_evaluation = generic_data_loading(config_data,case_type=case_type)

	for case_number in for_evaluation:
		if verbose>=50: print('    case number: %s'%(str(case_number)))
		x = for_evaluation[case_number][0] # shape example e.g. [1,6,19,192,12] 
		s_with_channel = x[0].shape # e.g. [6,19,192,12]
		s = s_with_channel[1:] # e.g. [19,192,192]
		if DEBUG_SHAPES_DURING_EVAL: print('      **DEBUG_SHAPES_DURING_EVAL:\n\t\tx.shape:%s'%(str(s)))
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
		y_pred = torch.argmax(this_net(x).contiguous(),dim=1).squeeze().permute(2,1,0).to(torch.float).cpu()
		if not np.all(y_pred.shape == s): 
			y_pred = interp3d(y_pred, s); print("",end='')
			if verbose>=50: print("      do reshape. y_pred.shape:%s [%s]"%(str(y_pred.shape),str(type(y_pred))))
		else:  
			if verbose>=50:  print("      no reshaping.") 

		ISLESDATA.save_one_case(y_pred, one_case, case_type, case_number,config_data, desc='etjoa001_UNet3D_test_'+str(case_number))
	# 	