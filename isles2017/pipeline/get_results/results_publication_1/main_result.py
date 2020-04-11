from utils.utils import * 

# DEBUG MODE?
# See utils/debug_switche

import pipeline.training as tr
import pipeline.evaluation as ev
import pipeline.lrp as lr
import pipeline.visual as vh

def example_process_get_main_figure(config_data):
	print('pipeline/get_results/restuls_publication_1.example_process_get_main_figure(). \
		You can hardcode parameters or config here.')

	# In the paper, other sizes are used as well.
	config_data['dataloader']['resize'] = [48,48,19]
	config_data['model_label_name'] = "UNet3D_AXXXS1" 
	
	tr.training_UNet3D(config_data)

	if not DEBUG_VERSION:
		config_data['LRP']['filter_sweeper']['case_numbers'] = [1,2,4,7,11,15,28,27,45]
	else:
		config_data['LRP']['filter_sweeper']['case_numbers'] = DEBUG_EVAL_TRAINING_CASE_NUMBERS
	
	ev.evaluation_UNet3D_overfit(config_data)
	lr.lrp_UNet3D_filter_sweeper_0002(config_data,verbose=0)
	lr.lrp_UNet3D_filter_sweeper_0003(config_data,verbose=0)
	config_data['LRP']['filter_sweeper']['submode'] = '0002'
	vh.lrp_UNet3D_filter_sweep_visualizer(config_data)
