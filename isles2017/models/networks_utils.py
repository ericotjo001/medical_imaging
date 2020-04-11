from utils.utils import *
import models.networks as net


def get_UNet3D_version(config_data, nc, training_mode=None, training=True):
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model')
	print("  get_UNet3D_version(). training_mode:%s"%(str(training_mode)))
	if training_mode == "UNet3D": 
		this_net = net.UNet3D(no_of_input_channel=nc, with_LRP=True); network_name = 'UNet3D'
	elif training_mode == "UNet3Db":
		this_net = net.UNet3Db(no_of_input_channel=nc, with_LRP=True); network_name = 'UNet3Db'
	elif training_mode == "UNet3D_diff":
		this_net = net.UNet3D(no_of_input_channel=nc, with_LRP=True); network_name = 'UNet3D'
	else:
		raise Exception('Invalid mode!')

	print('      ',main_model_fullpath)

	if training:
		if os.path.exists(main_model_fullpath): 
			this_net = this_net.load_state(config_data); 
			print("  Load existing model... [%s]"%(network_name))
		else: 
			print("  Creating new model... [%s]"%(network_name))
		this_net.training_cycle = this_net.training_cycle + 1
	else:
		
		if os.path.exists(main_model_fullpath): 
			this_net = this_net.load_state(config_data); 
			print("  Load existing model... [%s]"%(network_name))
		else: 
			raise Exception('  Model to load not found!') 

	return this_net