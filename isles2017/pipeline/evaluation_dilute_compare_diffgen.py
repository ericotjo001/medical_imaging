from utils.utils import *
from pipeline.evaluation import *
from dataio.data_diffgen import DG3D

# config_data['evaluation_mode'] == 'UNet3D_test_dilution'
# compare diffgen with dilution process.

pm = PrintingManager()
def eval_dilute_UNet_overfit_compare_diffgen(config_data,
	NUMBER_NOISE_TEST_PER_CASE=5,
	NUMBER_REPEAT_AVG=10, 
	DILUTE_TH=0.1,
	CASE_NUMBERS = range(1,49), 	 # missing number are 3,17,25,29
	case_type='training',
	verbose=0, tab_level=0):
	from utils.utils_dilution import FilterConv3D
	from pipeline.visual_dilute import show_dilute_results

	print('pipeline/evaluation_dilute_compare_diffgen.py .eval_dilute_UNet_overfit_compare_diffgen()')
	
	path_to_save_file = get_path_to_save_filename(config_data, filename='dilute.result')
	load_existing, dilute_result_dict = check_dilute_result_exist(path_to_save_file)

	if load_existing:
		show_dilute_results(dilute_result_dict)
		return
	else:
		dilute_result_dict['defect_test_data_by_case_number'] = {}
		dilute_result_dict['dilute_test_data_by_case_number'] = {}

	# conv_obj = torch.nn.Conv3d(1, 1, 7, stride=1, padding=3, bias=False)
	# conv_obj.weight.data = conv_obj.weight.data*0 + 1./len(conv_obj.weight.data.reshape(-1))		

	dg = DG3D(unit_size=tuple(config_data['dataloader']['resize'][:2]), depth=19) # this size will be interpolated to 3D (19,192,192)
	con_avg = FilterConv3D(conv_obj='average')
	con_avg.conv_obj.to(device=this_device)
	dice_loss = uloss.SoftDiceLoss()
	
	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
	model_type = config_data['training_mode']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])

	this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
	this_net.eval()
	for_evaluation = generic_data_loading(config_data, case_type=case_type, case_numbers_manual=CASE_NUMBERS)
	for case_number in for_evaluation:
		pm.printvm('case_number:%s'%(str(case_number)), tab_level=1, verbose=verbose, verbose_threshold=250)

		x = for_evaluation[case_number][0]
		s = x.shape[2:][::-1]

		outputs = torch.argmax(this_net(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs = interp3d(outputs,tuple(s), mode='nearest').detach()

		pm.printvm('x.shape:%s\nlabels.shape:%s\noutputs.shape:%s'%(str(x.shape),
			str(s), str(outputs.shape)), tab_level=2, verbose=verbose, verbose_threshold=250)

		dilute_result_dict['defect_test_data_by_case_number'][case_number] = []
		for i in range(NUMBER_NOISE_TEST_PER_CASE):
			outputs_defect, diff = get_defect_output(dg, this_net, x, s, config_data,  
				verbose=verbose, tab_level=tab_level+2)
			d = dice_loss(outputs, outputs_defect , factor=1)
			dice_score = 1 - d.item()
			dilute_result_dict['defect_test_data_by_case_number'][case_number].append({
				'diff': diff, 'dice_score': dice_score
			}) # [diff, dice_score]
			pm.printvm('%s'%(str(dice_score)), tab_level=3, verbose=verbose, verbose_threshold=250)
		
		dilute_result_dict['dilute_test_data_by_case_number'][case_number] = []
		x1 = x.clone().detach()
		for i in range(NUMBER_REPEAT_AVG):
			output_dilute, diff, x1 = get_dilution_output(con_avg, this_net, x, x1, s, config_data,  
				verbose=verbose, tab_level=tab_level+2)
			d = dice_loss(outputs, output_dilute , factor=1)
			dice_score = 1 - d.item()
			if dice_score < DILUTE_TH: break
			dilute_result_dict['dilute_test_data_by_case_number'][case_number].append(
				{'diff': diff, 'dice_score': dice_score})
			pm.printvm('%s'%(str(dice_score)), tab_level=3, verbose=verbose, verbose_threshold=250)
	save_dilute_data(path_to_save_file, dilute_result_dict)

def save_dilute_data(path_to_save_file, dilute_result_dict):
	output = open(path_to_save_file, 'wb')
	pickle.dump(dilute_result_dict, output)
	output.close()

def get_dilution_output(con_avg, this_net, x, x1, s, config_data,  
	verbose=0, tab_level=0):
	x1 = con_avg.channel_wise_conv(x1)
	output_dilute = torch.argmax(this_net(x1).contiguous(),dim=1)
	output_dilute = output_dilute.squeeze().permute(2,1,0).to(torch.float)
	output_dilute = interp3d(output_dilute,tuple(s), mode='nearest')

	diff = torch.sum(torch.abs(x-x1).reshape(-1))/len(x.clone().reshape(-1))
	pm.printvm('get_dilution_output(). diff:%s'%(str(diff.item())), tab_level=tab_level, verbose=verbose, verbose_threshold=250)
	
	return output_dilute, diff.item(), x1

def get_defect_output(dg, this_net, x, s, config_data, tab_level=0, verbose=0):
	# outputs_defect replaces x_unhealthy to save memory
	_, x_unhealthy, _, _ = dg.generate_data_batches_in_torch(
		channel_size=6, batch_size=1,resize=config_data['dataloader']['resize'])
	
	defect_fraction = np.random.uniform(0.5,2.4)

	x1 = x.clone().detach().to(torch.float) + defect_fraction * x_unhealthy.to(device=this_device).to(torch.float)
	x1 = x1.to(torch.float)

	diff = torch.sum(torch.abs(defect_fraction * x_unhealthy).reshape(-1))/len(x.clone().reshape(-1))

	pm.printvm('get_defect_output(). diff:%s'%(str(diff.item())), tab_level=tab_level, verbose=verbose, verbose_threshold=250)
	
	outputs_defect = torch.argmax(this_net(x1).contiguous(),dim=1)
	outputs_defect = outputs_defect.squeeze().permute(2,1,0).to(torch.float)
	outputs_defect = interp3d(outputs_defect,tuple(s), mode='nearest')
	return outputs_defect.detach(), diff.item()


def check_dilute_result_exist(path_to_save_file):
	load_existing = False
	if os.path.exists(path_to_save_file): 
		load_existing = True
		pkl_file = open(path_to_save_file, 'rb')
		dilute_result_dict = pickle.load(pkl_file)
		pkl_file.close()
	else:
		dilute_result_dict = {}

	return load_existing, dilute_result_dict

def get_path_to_save_filename(config_data, filename='dilute.result'):
	filepath = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'])
	filepath = os.path.join(filepath, config_data['model_label_name'], filename)
	return filepath