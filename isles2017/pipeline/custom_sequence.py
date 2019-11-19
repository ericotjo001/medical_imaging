from utils.utils import * 
import pipeline.training as tr
import pipeline.evaluation as ev
import pipeline.lrp as lr
import pipeline.visual as vi

"""
Custom sequence
Create your custom sequence in this script
"""

def full_sequence_0000(config_data):
	for i in range(2):
		tr.training_UNet3D(config_data)
	ev.evaluation_UNet3D_overfit(config_data)
	ev.evaluation_UNet3D_overfit_submission(config_data)
	ev.evaluation_UNet3D_test_submission(config_data)
	lr.lrp_UNet3D_overfit(config_data)

def full_sequence_0003(config_data):
	# model_names = ['UNet3D_AXXXS2']
	model_names = [
		# 'UNet3D_BXXXS1',
		# 'UNet3D_BXXXS2',
		'UNet3D_XXXXS5',
		'UNet3D_XXXXS6'
	]
	for model_name in model_names:
		config_data['model_label_name'] = model_name
		lr.lrp_UNet3D_filter_sweeper_0003(config_data,verbose=0)
	return

def full_sequence_0002(config_data, code):
	# this is for LRP sweeper submode 2
	size_code = str(code[0])
	trial_code = str(code[1])
	save_code = str(code[2]) # X means do not save, S means save 

	if size_code == 'A':
		config_data['dataloader']['resize'] = [48,48,19]
	elif size_code == 'B':
		config_data['dataloader']['resize'] = [96,96,19]
	elif size_code == 'C':
		config_data['dataloader']['resize'] = [144,144,19]
	elif size_code == 'X':
		config_data['dataloader']['resize'] = [192,192,19]

	if save_code == 'X':
		config_data['LRP']['filter_sweeper']['case_numbers'] = []
	elif save_code == 'S':
		config_data['LRP']['filter_sweeper']['case_numbers'] = [1,2,4,7,11,15,28,27,45]

	config_data['model_label_name'] = "UNet3D_" + size_code + "XXX" + save_code + trial_code
	
	tr.training_UNet3D(config_data)
	ev.evaluation_UNet3D_overfit(config_data)
	lr.lrp_UNet3D_filter_sweeper(config_data)


def full_sequence_0001(config_data, code):
	loop_number=2
	# config_modes = [
	# 	['UNet3D_XXXXXX','UNet3D'],
	# ]
	size_code = str(code[0])
	normalization_code = str(code[1:3])
	trial_code = str(code[3])
	config_data['model_label_name'] = "UNet3D_" + size_code + normalization_code + "XX" + trial_code
	
	if size_code == 'A':
		config_data['dataloader']['resize'] = [48,48,19]
	elif size_code == 'B':
		config_data['dataloader']['resize'] = [96,96,19]
	elif size_code == 'C':
		config_data['dataloader']['resize'] = [144,144,19]
	elif size_code == 'X':
		config_data['dataloader']['resize'] = [192,192,19]


	if normalization_code == 'XX':
		config_data['LRP']['relprop_config']['normalization'] = 'raw'
	elif normalization_code == 'Yp':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_pass_filter'
		config_data['LRP']['relprop_config']['fraction_pass_filter']['positive'] = [0.,6.]
		config_data['LRP']['relprop_config']['fraction_pass_filter']['negative'] = [-0.6,-0.]
	elif normalization_code == 'Zp':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_pass_filter'
		config_data['LRP']['relprop_config']['fraction_pass_filter']['positive'] = [0.,3.]
		config_data['LRP']['relprop_config']['fraction_pass_filter']['negative'] = [-0.3,-0.]
	elif normalization_code == 'Lp':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_pass_filter'
		config_data['LRP']['relprop_config']['fraction_pass_filter']['positive'] = [0.2,1.]
		config_data['LRP']['relprop_config']['fraction_pass_filter']['negative'] = [-1.,-0.2]
	elif normalization_code == 'Hp':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_pass_filter'
		config_data['LRP']['relprop_config']['fraction_pass_filter']['positive'] = [0.6,1.]
		config_data['LRP']['relprop_config']['fraction_pass_filter']['negative'] = [-1.,-0.6]
	elif normalization_code == 'Yc':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_clamp_filter'
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['positive'] = [0.0,0.6]
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['negative'] = [-0.6,-0.0]
	elif normalization_code == 'Zc':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_clamp_filter'
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['positive'] = [0.,3.]
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['negative'] = [-0.3,-0.]
	elif normalization_code == 'Lc':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_clamp_filter'
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['positive'] = [0.3,1.0]
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['negative'] = [-1.0,-0.3]
	elif normalization_code == 'Hc':
		config_data['LRP']['relprop_config']['normalization'] = 'fraction_clamp_filter'
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['positive'] = [0.6,1.0]
		config_data['LRP']['relprop_config']['fraction_clamp_filter']['negative'] = [-1.0,-0.6]
	else:
		raise Exception('Invalid mode')
	full_sequence_aux(config_data,loop_number)


def full_sequence_aux(config_data, loop_number):
	for i in range(loop_number):	
		# Training 
		tr.training_UNet3D(config_data)
		if DEBUG_TRAINING_LOOP: return

	# Evaluation and LRP
	ev.evaluation_UNet3D_overfit(config_data)
	# ev.evaluation_UNet3D_overfit_submission(config_data)
	# ev.evaluation_UNet3D_test_submission(config_data)
	lr.lrp_UNet3D_overfit(config_data)

	