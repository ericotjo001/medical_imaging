from utils.utils import * 
import pipeline.training as tr
import pipeline.evaluation as ev
import pipeline.lrp as lr
import pipeline.visual as vi

import pipeline.custom_sequence as cust

def print_info(config_dir):
	# print("raw_config_dir:%s"%(config_dir))
	print(DESCRIPTION)

def print_info_diff_gen(config_dir):
	print("raw_config_dir:%s"%(config_dir))
	print(DESCRIPTION_DIFF_GEN)

def create_config_file(config_dir):
	cm = ConfigManager()
	cm.create_config_file(config_dir)

	config_data = cm.json_file_to_pyobj(config_dir)
	cm.json_file_to_pyobj_recursive_print(config_data, name='config_data', verbose=0)
	config_data = cm.recursive_namedtuple_to_dict(config_data)

def train(config_data):	
	print("entry.py. train(). training mode = %s"%(str(config_data['training_mode'])))

	if config_data['training_mode'] == 'UNet3D':
		tr.training_UNet3D(config_data)
	elif config_data['training_mode'] == "UNet3D_diff": 
		import pipeline.training_UNet3D_diff as trd
		trd.training_UNet3D_diff(config_data)
	else: 
		print('Invalid mode specified.')
		from utils.description_training_modes import TRAINING_MODES_INFO
		print(TRAINING_MODES_INFO)
	
def evaluation(config_data):
	print("entry.py. evaluation().")
	
	if config_data['evaluation_mode'] == 'UNet3D_overfit': 
		ev.evaluation_UNet3D_overfit(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_overfit_submission': 
		ev.evaluation_UNet3D_overfit_submission(config_data)
	elif config_data['evaluation_mode'] == 'UNet3D_test_submission': 
		CASE_NUMBERS_SELECTION = range(1,41)
		for case_number in CASE_NUMBERS_SELECTION:
			case_numbers_manual = [case_number]
			ev.evaluation_UNet3D_test_submission(config_data,case_numbers_manual=case_numbers_manual)
	elif config_data['evaluation_mode'] == 'UNet3D_test_dilution': 
		from pipeline.evaluation_dilute import select_mode_UNet3D_test_dilution
		select_mode_UNet3D_test_dilution(config_data)
	else:
		print('Invalid mode specified.') 
		from utils.description_eval_modes import EVAL_MODES_INFO
		print(EVAL_MODES_INFO)
	
		

def lrp(config_data):
	print("entry.py. lrp().")

	if config_data['lrp_mode'] == 'lrp_UNet3D_overfit': 
		lr.lrp_UNet3D_overfit(config_data)
	elif config_data['lrp_mode'] == 'lrp_UNet3D_filter_sweeper': 
		lr.lrp_UNet3D_filter_sweeper(config_data)
	else: 
		raise Exception('Invalid mode specified.')


def visual(config_data):
	print("entry.py. visual()")
	vi.visual_select_submode(config_data)

def shortcut_sequence(config_data, mode=None):
	"""
	Adhoc sequences for smoother processes.
	Do anything custom edits here.
	"""
	from datetime import date
	from datetime import datetime
	today = date.today()
	d1 = today.strftime("%d/%m/%Y")
	d2 = today.strftime("%B %d, %Y")
	now = datetime.now()
	dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

	model_dir = os.path.join(config_data["working_dir"],config_data["relative_checkpoint_dir"])
	if not os.path.exists(model_dir): os.mkdir(model_dir)
	full_path_log_file = os.path.join(model_dir,'logfile_'+str(mode)+'_'+str(dt_string)+'.txt')
	sys.stdout = Logger(full_path_log_file=full_path_log_file)


	start = time.time()

	if mode is None:
		print("No mode specified. Terminating... See available modes in pipeline/entry.py")
	elif mode == 'diff_gen_shortcut':
		print('running diff_gen_shortcut mode')
		import pipeline.custom_sequence2 as cs2
		cs2.diffgen_sequence(config_data)
	else:
		print('Running standard full sequence using the code %s'%(str(mode)))
		# UNCOMMENT THE MODES YOU WANT HERE
		# cust.full_sequence_0002(config_data, mode)
		cust.full_sequence_0003(config_data, )

	end = time.time()
	elapsed = end - start
	elapsed_min = elapsed/60.
	elapsed_hr = elapsed_min/60.
	print('time taken [s]:%s'%(str(elapsed))) 
	print('time taken [min]:%s'%(str(elapsed_min)))
	print('time taken [hr]:%s'%(str(elapsed_hr)))
