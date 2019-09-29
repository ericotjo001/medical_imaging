from utils.utils import * 
import pipeline.training as tr
import pipeline.evaluation as ev
import pipeline.lrp as lr

def print_info(config_dir):
	print("raw_config_dir:%s"%(config_dir))
	print(DESCRIPTION)

def create_config_file(config_dir):
	cm = ConfigManager()
	cm.create_config_file(config_dir)

	config_data = cm.json_file_to_pyobj(config_dir)
	cm.json_file_to_pyobj_recursive_print(config_data, name='config_data', verbose=0)
	config_data = cm.recursive_namedtuple_to_dict(config_data)

def train(config_dir):
	print("entry.py. train().")

	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)

	if config_data['training_mode'] == 'UNet3D': tr.training_UNet3D(config_data)
	# elif config_raw_data['training_mode'] == 'FCN8like': tr.training_FCN8like(config_data)
	# elif config_raw_data['training_mode'] == 'PSPNet': tr.training_PSPNet(config_data)
	# elif config_raw_data['training_mode'] == 'segnet': tr.training_segnet(config_data)
	else: raise Exception('Invalid mode specified.')
	
def evaluation(config_dir):
	print("entry.py. evaluation().")
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)
	
	if config_data['evaluation_mode'] == 'UNet3D_overfit': ev.evaluation_UNet3D_overfit(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_overfit_submission': ev.evaluation_UNet3D_overfit_submission(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_test_submission': ev.evaluation_UNet3D_test_submission(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_eval_combo': 
		ev.evaluation_UNet3D_overfit(config_data)
		ev.evaluation_UNet3D_overfit_submission(config_data)
		ev.evaluation_UNet3D_test_submission(config_data)
	# elif config_data['evaluation_mode'] == 'FCN_1_overfit': ev.evaluation_FCN_1_overfit(config_data)
	# elif config_data['evaluation_mode'] == 'FCN8like_overfit': ev.evaluation_FCN8like_overfit(config_data)
	# elif config_data['evaluation_mode'] == 'PSPNet_overfit': ev.evaluation_PSPNet_overfit(config_data)
	# elif config_data['evaluation_mode'] == 'segnet_overfit': ev.evaluation_segnet_overfit(config_data)
	else: raise Exception('Invalid mode specified.')

def lrp(config_dir):
	print("entry.py. lrp().")
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)

	if config_data['lrp_mode'] == 'lrp_UNet3D_overfit': lr.lrp_UNet3D_overfit(config_data)
	elif config_data['lrp_mode'] == 'lrp_UNet3D_overfit_visualizer': lr.lrp_UNet3D_overfit_visualizer(config_data)
	else: raise Exception('Invalid mode specified.')

def shortcut_sequence(config_dir):
	"""
	Adhoc sequences for smoother processes.
	Do anything custom edits here.
	"""
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)

	# Training phase 1
	if config_data['training_mode'] == 'UNet3D': tr.training_UNet3D(config_data)
	else: raise Exception('Invalid mode specified.')

	# Training phase 2
	config_data['basic']['n_epoch']	= 10
	config_data['augmentation']['type'] = 'rotate_then_clip_translate'
	config_data['augmentation']['number_of_data_augmented_per_case'] = 10
	if config_data['training_mode'] == 'UNet3D': tr.training_UNet3D(config_data)
	else: raise Exception('Invalid mode specified.')

	# Evaluation
	if config_data['evaluation_mode'] == 'UNet3D_eval_combo': 
		ev.evaluation_UNet3D_overfit(config_data)
		ev.evaluation_UNet3D_overfit_submission(config_data)
		ev.evaluation_UNet3D_test_submission(config_data)
	else: raise Exception('Invalid mode specified.')
	
	# LRP
	if config_data['lrp_mode'] == 'lrp_UNet3D_overfit': lr.lrp_UNet3D_overfit(config_data)
	elif config_data['lrp_mode'] == 'lrp_UNet3D_overfit_visualizer': lr.lrp_UNet3D_overfit_visualizer(config_data)
	else: raise Exception('Invalid mode specified.')
	