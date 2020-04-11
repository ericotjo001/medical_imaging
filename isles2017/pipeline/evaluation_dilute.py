from utils.utils import *

DILUTE_INFO = """Evaluation. Dilution mode availables:

  python main.py --mode evaluation --submode UNet3D_overfit_compare_diffgen
  python main.py --mode evaluation --submode UNet3D_overfit_more_comparison
"""


def select_mode_UNet3D_test_dilution(config_data):
	print('select_mode_UNet3D_test_dilution(). submode:%s'%(str(config_data['console_submode'])))

	if config_data['console_submode'] is None:
		print(DILUTE_INFO)
	elif config_data['console_submode'] == 'UNet3D_overfit_compare_diffgen':
		import pipeline.evaluation_dilute_compare_diffgen as cd
		if DEBUG_VERSION:
			CASE_NUMBERS = range(1,2,5)
			NUMBER_NOISE_TEST_PER_CASE = 4
			NUMBER_REPEAT_AVG = 10
		else:
			CASE_NUMBERS = range(1,49)# missing number for training case are 3,17,25,29
			NUMBER_NOISE_TEST_PER_CASE = 50
			NUMBER_REPEAT_AVG = 24

		cd.eval_dilute_UNet_overfit_compare_diffgen(config_data,
			NUMBER_NOISE_TEST_PER_CASE=NUMBER_NOISE_TEST_PER_CASE,
			NUMBER_REPEAT_AVG=NUMBER_REPEAT_AVG, 
			DILUTE_TH=0.01,
			CASE_NUMBERS = CASE_NUMBERS,
			case_type='training',
			verbose=250, tab_level=0)

	elif config_data['console_submode'] == 'UNet3D_overfit_more_comparison':
		if DEBUG_VERSION:
			CASE_NUMBERS = range(1,8)
			NUMBER_NOISE_TEST_PER_CASE = 4
		else:
			CASE_NUMBERS = range(1,49)
			NUMBER_NOISE_TEST_PER_CASE = 50

		import pipeline.evaluation_dilute_compare_diffgen2 as cd
		cd.eval_other_UNet_overfit_compare_diffgen(config_data, 
			NUMBER_NOISE_TEST_PER_CASE=NUMBER_NOISE_TEST_PER_CASE,
			CASE_NUMBERS = CASE_NUMBERS, 	 
			case_type='training',
			gaussian_mean=0.,
			gaussian_sd=0.15,
			zeroing_fraction=0.5,
			verbose=250, tab_level=0)
	else:
		print("Invalid mode!")
		print(DILUTE_INFO)