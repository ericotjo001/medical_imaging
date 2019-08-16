from utils.utils import * 
import pipeline.training as tr
import pipeline.evaluation as ev

def print_info(config_dir):
	print("raw_config_dir:%s"%(config_dir))
	print(DESCRIPTION)

def create_config_file(config_dir):
	with open(config_dir, 'w') as json_file:  
		json.dump(CONFIG_FILE, json_file, separators=(',', ': '), indent=2)
	'''loading'''
	config_raw_data = json_to_dict(config_dir)
	print("create_config_file(). Config file created.")
	config_data = prepare_config(config_raw_data)
	printing_config(config_data)

def train(config_dir):
	print("entry.py. train().")
	config_raw_data = json_to_dict(config_dir)
	config_data = prepare_config(config_raw_data)
	if config_data['training_mode'] == 'basic_1': tr.training_basic_1(config_data)
	# elif config_raw_data['training_mode'] == 'FCN_1': tr.training_FCN_1(config_data)
	elif config_raw_data['training_mode'] == 'FCN8like': tr.training_FCN8like(config_data)
	elif config_raw_data['training_mode'] == 'UNet3D': tr.training_UNet3D(config_data)
	elif config_raw_data['training_mode'] == 'PSPNet': tr.training_PSPNet(config_data)
	elif config_raw_data['training_mode'] == 'segnet': tr.training_segnet(config_data)
	else: raise Exception('Invalid mode specified.')
	
def evaluation(config_dir):
	print("entry.py. evaluation().")
	config_raw_data = json_to_dict(config_dir)
	config_data = prepare_config(config_raw_data)
	if config_data['evaluation_mode'] == 'basic_1_overfit': ev.evaluation_basic_1_overfit(config_data)
	# elif config_data['evaluation_mode'] == 'FCN_1_overfit': ev.evaluation_FCN_1_overfit(config_data)
	elif config_data['evaluation_mode'] == 'FCN8like_overfit': ev.evaluation_FCN8like_overfit(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_overfit': ev.evaluation_UNet3D_overfit(config_data)
	elif config_data['evaluation_mode'] == 'PSPNet_overfit': ev.evaluation_PSPNet_overfit(config_data)
	elif config_data['evaluation_mode'] == 'segnet_overfit': ev.evaluation_segnet_overfit(config_data)
	else: raise Exception('Invalid mode specified.')