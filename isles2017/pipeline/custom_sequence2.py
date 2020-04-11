from utils.utils import * 
import pipeline.training as tr
import pipeline.evaluation as ev
# import pipeline.lrp as lr
# import pipeline.visual as vi

CUSTOM_SEQUENCE2_INFO = """** Avaliable CUSTOM_SEQUENCE2 modes:
python main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut --submode 0001
python main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut --submode 0002
python main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut --submode 0003
"""

def diffgen_sequence(config_data):
	print('diffgen_sequence. submode:%s'%(str(config_data['console_submode'])))
	if config_data['console_submode'] is None:
		print(CUSTOM_SEQUENCE2_INFO)
	elif config_data['console_submode'][:4] == str('0001'):
		if config_data['console_submode'] == str('0001'):
			print('\nNow, set the 4-digit code for console_submode, for example:')
			print('\npython main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut --submode 0001.XXXX\n')
			return
		diffgen_sequence_0001(config_data)
	elif config_data['console_submode'][:4] == str('0002'):
		if config_data['console_submode'] == str('0002'):
			print('\nNow, set the 4-digit code for console_submode, for example:')
			print('\npython main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut --submode 0002.XXXX\n')
			return
		diffgen_sequence_0002(config_data)	
	elif config_data['console_submode'][:4] == str('0003'):
		if config_data['console_submode'] == str('0003'):
			print('\nNow, set the 4-digit code for console_submode, for example:')
			print('\npython main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut --submode 0003.XXXX\n')
			return
		diffgen_sequence_0003(config_data)	
	else:
		print('invalid mode selected')
		print(CUSTOM_SEQUENCE2_INFO)


def diffgen_sequence_0001(config_data):
	code = config_data['console_submode'][5:]
	print('diffgen_sequence_0001(). code:%s'%(str(code)))
	# example
	# python main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut --submode 0001.SX11

	size_code = code[0]
	if size_code=='X': config_data['dataloader']['resize'] = [192,192,19]
	if size_code=='S': config_data['dataloader']['resize'] = [48,48,19]

	epoch_size = code[1]
	config_data['basic']['n_epoch'] = 1

	DATA_PER_EPOCH = 1250
	NO_OF_RUNS = 2

	if DEBUG_VERSION:
		DATA_PER_EPOCH = 2
		NO_OF_RUNS = 2

	fraction_code = code[2] # int
	if str(fraction_code) == str(1):
		defect_fraction = 1
	elif str(fraction_code) == str(2):
		defect_fraction = 1.2
		
	name_code = code[3]
	config_data['model_label_name'] = "UNet3D_"+str(size_code)+"DGX" + str(fraction_code) + str(name_code)

	pm = PrintingManager()
	print('Configs:')
	pm.print_recursive_dict(config_data ,tab_level=2, tab_shape='  ',verbose=0, verbose_threshold=None)

	import pipeline.evaluation as ev
	import pipeline.training_UNet3D_diff as trd
	for i in range(NO_OF_RUNS):
		trd.training_UNet3D_diff(config_data, DATA_PER_EPOCH,defect_fraction=defect_fraction)
		print('\nRUN:%s training completed.\n'%(str(i)))
	ev.generic_evaluation_overfit_0001(config_data)

	print()
	print("="*80)
	print()

def diffgen_sequence_0002(config_data):
	if DEBUG_VERSION:
		CASE_NUMBERS = range(1,2,5)
		NUMBER_NOISE_TEST_PER_CASE = 4
		NUMBER_REPEAT_AVG = 10
	else:
		CASE_NUMBERS = range(1,49)# missing number for training case are 3,17,25,29
		NUMBER_NOISE_TEST_PER_CASE = 50
		NUMBER_REPEAT_AVG = 24
		
	code = config_data['console_submode'][5:]
	print('diffgen_sequence_0002(). code:%s'%(str(code)))	

	custom_code = code[0]
	# freely customize here

	if custom_code =='X':
		config_data['model_label_name'] = 'UNet3D_XDGX22'
		config_data['dataloader']['resize'] = [192,192,19]
	elif custom_code == 'S':
		config_data['model_label_name'] = 'UNet3D_SDGX21'
		config_data['dataloader']['resize'] = [48,48,19]

	import pipeline.evaluation_dilute_compare_diffgen as cd

	cd.eval_dilute_UNet_overfit_compare_diffgen(config_data,
		NUMBER_NOISE_TEST_PER_CASE=50,
		NUMBER_REPEAT_AVG=24, 
		DILUTE_TH=0.01,
		CASE_NUMBERS = range(1,49), 	 # missing number are 3,17,25,29
		case_type='training',
		verbose=250, tab_level=0)

def diffgen_sequence_0003(config_data):

	if DEBUG_VERSION:
		CASE_NUMBERS = range(1,8)
		NUMBER_NOISE_TEST_PER_CASE = 4
	else:
		CASE_NUMBERS = range(1,49)
		NUMBER_NOISE_TEST_PER_CASE = 50

	code = config_data['console_submode'][5:]
	print('diffgen_sequence_0003(). code:%s'%(str(code)))	

	custom_code = code[0]
	# freely customize here

	if custom_code =='X':
		config_data['model_label_name'] = 'UNet3D_XDGX22'
		config_data['dataloader']['resize'] = [192,192,19]
	elif custom_code == 'S':
		config_data['model_label_name'] = 'UNet3D_SDGX21'
		config_data['dataloader']['resize'] = [48,48,19]

	import pipeline.evaluation_dilute_compare_diffgen2 as cd
	cd.eval_other_UNet_overfit_compare_diffgen(config_data, 
		NUMBER_NOISE_TEST_PER_CASE=NUMBER_NOISE_TEST_PER_CASE,
		CASE_NUMBERS = CASE_NUMBERS, 	 
		case_type='training',
		gaussian_mean=0.,
		gaussian_sd=0.15,
		zeroing_fraction=0.5,
		verbose=250, tab_level=0)