from utils.utils import * 
import pipeline.training as tr
import pipeline.evaluation as ev
import pipeline.lrp as lr
import pipeline.visual as vi

import pipeline.custom_sequence as cust

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
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)
	
	print("entry.py. train(). training mode = %s"%(str(config_data['training_mode'])))

	if config_data['training_mode'] == 'UNet3D': tr.training_UNet3D(config_data)
	elif config_data['training_mode'] == 'UNet3D_LRP_optim': tr.training_UNet3D(config_data)
	else: raise Exception('Invalid mode specified.')
	
def evaluation(config_dir):
	print("entry.py. evaluation().")
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)
	
	if config_data['evaluation_mode'] == 'UNet3D_overfit': ev.evaluation_UNet3D_overfit(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_overfit_submission': ev.evaluation_UNet3D_overfit_submission(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_test_submission': ev.evaluation_UNet3D_test_submission(config_data)
	else: raise Exception('Invalid mode specified.')

def lrp(config_dir):
	print("entry.py. lrp().")
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)

	if config_data['lrp_mode'] == 'lrp_UNet3D_overfit': lr.lrp_UNet3D_overfit(config_data)
	elif config_data['lrp_mode'] == 'lrp_UNet3D_filter_sweeper': lr.lrp_UNet3D_filter_sweeper(config_data)
	else: raise Exception('Invalid mode specified.')


def visual(config_dir):
	print("entry.py. visual()")
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)
	
	vi.visual_select_submode(config_data)

def shortcut_sequence(config_dir, mode=None):
	"""
	Adhoc sequences for smoother processes.
	Do anything custom edits here.
	"""
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)
	
	model_dir = os.path.join(config_data["working_dir"],config_data["relative_checkpoint_dir"])
	if not os.path.exists(model_dir): os.mkdir(model_dir)
	full_path_log_file = os.path.join(model_dir,'logfile_'+str(mode)+'.txt')
	sys.stdout = Logger(full_path_log_file=full_path_log_file)


	start = time.time()

	if mode is None:
		print("No mode specified")
	else:
		# cust.full_sequence_0002(config_data, mode)
		cust.full_sequence_0003(config_data, )
	end = time.time()
	elapsed = end - start
	elapsed_min = elapsed/60.
	elapsed_hr = elapsed_min/60.
	print('time taken [s]:%s'%(str(elapsed))) 
	print('time taken [min]:%s'%(str(elapsed_min)))
	print('time taken [hr]:%s'%(str(elapsed_hr)))
