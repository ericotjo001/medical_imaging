from utils.utils import * 
import pipeline.training as tr
import pipeline.evaluation as ev

def train(config_dir):
	print("entry.py. train().")
	config_data = get_config_data(config_dir)

	if config_data['training_mode'] == 'training_small_cnn': tr.training_small_cnn(config_data)
	elif config_data['training_mode'] == 'training_sr_small_cnn': tr.training_sr_small_cnn(config_data)
	else: raise Exception('Mode invalid')

def evaluation(config_dir):
	print("entry.py. evaluation().")
	config_data = get_config_data(config_dir)

	if config_data['evaluation_mode'] ==  'evaluate_small_cnn': ev.evaluate_small_cnn(config_data)
	else: raise Exception('Mode invalid')

def shortcut_sequence(config_dir):
	print("entry.py. shortcut_sequence().")
	config_data = get_config_data(config_dir)

	'''
	Customize your sequence here!
	'''
	for i in range(4):
		if config_data['training_mode'] == 'training_small_cnn': tr.training_small_cnn(config_data)
		elif config_data['training_mode'] == 'training_sr_small_cnn': tr.training_sr_small_cnn(config_data)
		else: raise Exception('Mode invalid')

		if config_data['evaluation_mode'] ==  'evaluate_small_cnn': ev.evaluate_small_cnn(config_data)
		else: raise Exception('Mode invalid')