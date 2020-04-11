from utils.utils import *
from pipeline.evaluation import *

# config_data['evaluation_mode'] == 'UNet3D_test_dilution'
# extension to evaluation_dilute_compare_diffgen.py
#   i.e. compare diffgen with other noisy methods

pm = PrintingManager()
def eval_other_UNet_overfit_compare_diffgen(config_data, 
	NUMBER_NOISE_TEST_PER_CASE=10,
	CASE_NUMBERS = range(1,49), 	 # missing number are 3,17,25,29
	case_type='training',
	gaussian_mean=0.,
	gaussian_sd=1.,
	zeroing_fraction=0.8,
	verbose=0, tab_level=0):
	pm.printvm('pipeline/evaluation_dilute_compare_diffgen.py. eval_other_UNet_overfit_compare_diffgen()', tab_level=tab_level)

	from pipeline.evaluation_dilute_compare_diffgen import get_path_to_save_filename, \
		check_dilute_result_exist, save_dilute_data
	from pipeline.visual_dilute2 import show_compare_other_results

	path_to_save_file = get_path_to_save_filename(config_data, filename='dilute_compare.result')
	load_existing, dilute_result_dict = check_dilute_result_exist(path_to_save_file)

	if load_existing:
		show_compare_other_results(dilute_result_dict)
		return
	else:
		pm.printvm('Creating new...', tab_level=tab_level+1)
		dilute_result_dict['gaussian_test_data_by_case_number'] = {}
		dilute_result_dict['zeroing_test_data_by_case_number'] = {}
		
	dice_loss = uloss.SoftDiceLoss()
	
	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
	model_type = config_data['training_mode']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])

	this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
	this_net.eval()
	for_evaluation = generic_data_loading(config_data, case_type=case_type, case_numbers_manual=CASE_NUMBERS)
	for case_number in for_evaluation:
		pm.printvm('case_number:%s'%(str(case_number)), tab_level=tab_level+1, verbose=verbose, verbose_threshold=250)

		x = for_evaluation[case_number][0]
		s = x.shape[2:][::-1]

		outputs = torch.argmax(this_net(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs = interp3d(outputs,tuple(s), mode='nearest').detach()

		pm.printvm('SaltPepper'%(), tab_level=2, verbose=verbose, verbose_threshold=250)		
		dilute_result_dict['zeroing_test_data_by_case_number'][case_number] = []
		for i in range(NUMBER_NOISE_TEST_PER_CASE):
			outputs_defect, diff = get_pixelwise_zeroing_output(this_net, x, s, config_data, fraction_min=zeroing_fraction)
			d = dice_loss(outputs, outputs_defect , factor=1)
			dice_score = 1 - d.item()
			dilute_result_dict['zeroing_test_data_by_case_number'][case_number].append(
				{'diff': diff, 'dice_score': dice_score})
			pm.printvm('diff: %s dice: %s'%(str(diff),str(dice_score)), tab_level=3, verbose=verbose, verbose_threshold=250)
		
		pm.printvm('Ugaussian'%(), tab_level=2, verbose=verbose, verbose_threshold=250)		
		dilute_result_dict['gaussian_test_data_by_case_number'][case_number] = []
		for i in range(NUMBER_NOISE_TEST_PER_CASE):
			outputs_defect, diff = get_gaussian_noised_output(this_net, x, s, config_data, mean=gaussian_mean, sd=gaussian_sd)
			d = dice_loss(outputs, outputs_defect , factor=1)
			dice_score = 1 - d.item()
			dilute_result_dict['gaussian_test_data_by_case_number'][case_number].append(
				{'diff': diff, 'dice_score': dice_score})
			pm.printvm('diff: %s dice: %s'%(str(diff),str(dice_score)), tab_level=3, verbose=verbose, verbose_threshold=250)
	save_dilute_data(path_to_save_file, dilute_result_dict)

def get_pixelwise_zeroing_output(this_net, x, s, config_data, fraction_min=0.8):
	fraction = np.random.uniform(fraction_min, 1.)
	coin_toss = np.random.uniform(0., 1., size=list(x.shape))
	coin_toss = torch.tensor(coin_toss <= fraction).to(torch.float)
	
	# for checking
	# leng = len(coin_toss.clone().detach().reshape(-1))
	# total_sum = torch.sum(coin_toss.reshape(-1))
	# fraction_non_zeroed = total_sum/leng
	# print(fraction_non_zeroed) # should be around 1-fraction

	x1 = x.clone().detach().to(torch.float) * coin_toss.to(device=this_device)
	diff = torch.sum(torch.abs(x-x1).reshape(-1))/len(x.clone().reshape(-1))

	outputs_defect = torch.argmax(this_net(x1).contiguous(),dim=1)
	outputs_defect = outputs_defect.squeeze().permute(2,1,0).to(torch.float)
	outputs_defect = interp3d(outputs_defect,tuple(s), mode='nearest').detach()
	return outputs_defect.detach(), diff.item()


def get_gaussian_noised_output(this_net, x, s, config_data, mean=0, sd=0.1):
	this_max = np.abs(np.random.normal(0,sd))
	this_noise = np.random.uniform(0, this_max, size=list(x.shape))
	this_noise = torch.tensor(this_noise).to(device=this_device).to(torch.float)
	
	diff = torch.sum(this_noise.reshape(-1))/len(x.clone().reshape(-1))

	x1 = x.clone().detach().to(torch.float) + this_noise
	outputs_defect = torch.argmax(this_net(x1).contiguous(),dim=1)
	outputs_defect = outputs_defect.squeeze().permute(2,1,0).to(torch.float)
	outputs_defect = interp3d(outputs_defect,tuple(s), mode='nearest').detach()
	
	return outputs_defect.detach(), diff.item()

