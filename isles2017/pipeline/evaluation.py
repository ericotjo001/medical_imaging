from pipeline.evaluation_header import *

def evaluation_UNet3D_overfit(config_data):
	print("evaluation_UNet3D_overfit()")
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	model_type = 'UNet3D'
	generic_evaluation_overfit_0001(config_data,model_type,model_dir)

def evaluation_UNet3D_overfit_submission(config_data):
	print("evaluation_UNet3D_overfit_submission()")
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	model_type = 'UNet3D'
	evaluation_for_submission_overfit_0001(config_data, model_type, model_dir)

def evaluation_UNet3D_test_submission(config_data):
	print('evaluation_UNet3D_test_submission()')
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	model_type = 'UNet3D'
	evaluation_for_test_submission_0001(config_data, model_type, model_dir)

def get_modalities_0001(config_data):
	modalities_dict = {}
	for i, mod in enumerate(config_data['data_modalities']): 
		if mod != 'OT': modalities_dict[i] = mod
	no_of_input_channel = len(modalities_dict)
	print("no_of_input_channel:%s"%(str(no_of_input_channel)))
	for xkey in modalities_dict: print("  %s"%(str(modalities_dict[xkey])))
	return modalities_dict, no_of_input_channel


'''
More implementation details
'''

def generic_evaluation_overfit_0001(config_data, model_type,model_dir):
	'''
	Assume OT is loaded
	Assume PWI is not included.
	'''
	ev = EvalObj()
	ev2 = EvalObj() # anotehr dice computation
	ev3 = EvalObj() # same as ev, but without eval()

	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)

	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	if model_type == 'UNet3D': this_net = net.UNet3D(no_of_input_channel=no_of_input_channels) 
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")

	for_evaluation = generic_data_loading(config_data)

	for save_epoch in this_net.saved_epochs:
		if model_type == 'UNet3D': artifact_net = net.UNet3D(no_of_input_channel=no_of_input_channels) 
		artifact_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(save_epoch) + '.model')
		if not os.path.exists(artifact_model_fullpath): continue
		artifact_net.load_state_dict(torch.load(artifact_model_fullpath))
		
		print("Looking at model after epoch %s"%(str(save_epoch)))
		generic_evaluation_overfit_0001_aux(artifact_net,for_evaluation,config_data, ev, ev2, ev3, eval_artifact=True, artifact_epoch=save_epoch)

	if DEBUG_EVAL_LOOP: return

	print("Looking at the latest model.")
	generic_evaluation_overfit_0001_aux(this_net,for_evaluation,config_data, ev, ev2, ev3, eval_artifact=False)

def evaluation_for_submission_overfit_0001(config_data, model_type, model_dir):
	'''
	This evaluation is compatible with training using data loaded from
	- ISLES2017mass() load_many_cases_type0003().
	'''
	print("evaluation_for_submission_overfit_0001()")
	ev = EvalObj()

	modalities_dict, no_of_input_channel = get_modalities_0001(config_data)
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	if model_type == 'UNet3D': this_net = net.UNet3D(no_of_input_channel=no_of_input_channel)

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	resize_shape = tuple(config_data['dataloader']['resize'])
	
	case_type = 'training'
	case_numbers = range(1,49)
	if DEBUG_EVAL_TRAINING_CASE_NUMBERS is not None: case_numbers = DEBUG_EVAL_TRAINING_CASE_NUMBERS

	processed_case_numbers = []
	for case_number in case_numbers:
		print("case_number:",case_number)
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)

		if one_case is None: continue
		else: processed_case_numbers.append(case_number)

		x1, y_ot = ISLESDATA.load_type0003_prepare_data_point(one_case, no_of_input_channel, resize_shape, 
				modalities_dict, config_data)
		
		x_in_input_shape = torch.tensor(x1.reshape((1,) + (no_of_input_channel,) + resize_shape))
		x_in_input_shape = x_in_input_shape.permute(0,1,4,3,2).to(torch.float).to(device=this_device)
		# print("x_in_input_shape.shape=%s CHECK=%s"%(str(x_in_input_shape.shape), \
		# 	str(np.all(x_in_input_shape.numpy()[0,:,:,:,:] == x1.transpose(0,3,2,1)))))

		y_pred = torch.argmax(this_net(x_in_input_shape).contiguous(),dim=1).squeeze().permute(2,1,0).to(torch.float).cpu()
		if not np.all(y_pred.shape == y_ot.shape): 
			y_pred = interp3d(y_pred, s); print("  do reshape.")
		else: print("  no reshaping.") 
		print("  y_pred.shape:%s [%s]"%(str(y_pred.shape),str(type(y_pred))))
		print("  y_ot.shape:%s [%s]"%(str(y_ot.shape),str(type(y_ot))))

		ISLESDATA.save_one_case(y_pred, one_case, case_type, case_number,config_data, desc='etjoa001_UNet3D_'+str(case_number))
		ev.save_one_case_evaluation(case_number, y_pred, torch.tensor(y_ot), config_data, dice=True)
	print('processed_case_numbers:%s'%(processed_case_numbers))

def evaluation_for_test_submission_0001(config_data, model_type, model_dir):
	print('evaluation_UNet3D_test_submission()')
	modalities_dict, no_of_input_channel = get_modalities_0001(config_data)
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	if model_type == 'UNet3D': this_net = net.UNet3D(no_of_input_channel=no_of_input_channel)

	ISLESDATA = ISLES2017mass()
	ISLESDATA.verbose = 0
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	resize_shape = tuple(config_data['dataloader']['resize'])

	case_numbers = range(1,41)
	case_type = 'test'
	if DEBUG_EVAL_TEST_CASE_NUMBERS is not None: case_numbers = DEBUG_EVAL_TEST_CASE_NUMBERS

	processed_case_numbers = []
	for case_number in case_numbers:
		print("case_number:",case_number)
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)

		if one_case is None: continue
		else: processed_case_numbers.append(case_number)

		first_key = list(one_case['imgobj'].keys())[0]; s = one_case['imgobj'][first_key].shape
		x1 = ISLESDATA.load_type0003_prepare_data_point(one_case, no_of_input_channel, resize_shape, 
				modalities_dict, config_data, mode=case_type)
		x_in_input_shape = torch.tensor(x1.reshape((1,) + (no_of_input_channel,) + resize_shape))
		x_in_input_shape = x_in_input_shape.permute(0,1,4,3,2).to(torch.float).to(device=this_device)
		
		# ISLESDATA.save_one_case(y_pred, one_case, case_type, case_number,config_data, desc='etjoa001_UNet3D_'+str(case_number))
		
		y_pred = torch.argmax(this_net(x_in_input_shape).contiguous(),dim=1).squeeze().permute(2,1,0).to(torch.float).cpu()
		if not np.all(y_pred.shape == s): 
			y_pred = interp3d(y_pred, s); print("  do reshape.")
		else: print("  no reshaping.") 
		print("  y_pred.shape:%s [%s]"%(str(y_pred.shape),str(type(y_pred))))

		ISLESDATA.save_one_case(y_pred, one_case, case_type, case_number,config_data, desc='etjoa001_UNet3D_test_'+str(case_number))
		