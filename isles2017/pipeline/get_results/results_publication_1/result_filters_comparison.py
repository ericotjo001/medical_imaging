from utils.utils import * 

# DEBUG MODE?
# See utils/debug_switche

import pipeline.training as tr
import pipeline.evaluation as ev
import pipeline.lrp as lr
import pipeline.visual_header as vh

def example_process_get_filters_comparison_figure(config_data):
	print('pipeline/get_results/restuls_publication_1.example_process_get_filters_comparison_figure(). \
		You can hardcode parameters or config here.')

	# The following is a mini-example
	# In the publication, larger sizes up to [192,192,19] and more runs are specified
	size_and_series_name = [
		([48,48,19], "UNet3D_AXXXS1"),
		([48,48,19], "UNet3D_AXXXS2"),
		([96,96,19], "UNet3D_BXXXS1"),
	]

	for this_resize, this_name in size_and_series_name:
		config_data['dataloader']['resize'] = this_resize
		config_data['model_label_name'] = this_name 
		
		tr.training_UNet3D(config_data)
		if not DEBUG_VERSION:
			config_data['LRP']['filter_sweeper']['case_numbers'] = [1,2,4,7,11,15,28,27,45]
		else:
			config_data['LRP']['filter_sweeper']['case_numbers'] = DEBUG_EVAL_TRAINING_CASE_NUMBERS
		
		ev.evaluation_UNet3D_overfit(config_data)
		lr.lrp_UNet3D_filter_sweeper_0002(config_data,verbose=0)
		lr.lrp_UNet3D_filter_sweeper_0003(config_data,verbose=0)

	# see size_and_series_name above
	model_label_names_collection = { 
		'[48,48,19]': ["UNet3D_AXXXS1", "UNet3D_AXXXS2"],
		'[96,96,19]': [ "UNet3D_BXXXS1"]
	}
	
	color_list = {
		'[48,48,19]': 'g',
		'[96,96,19]': 'r',
	}
	vh.lrp_UNet3D_filter_sweep_0002_visualizer_aux(config_data,model_label_names_collection, 'fraction_pass_filter', color_list, verbose=0)
	print("\n")
	print("="*100)
	print("\n")
	vh.lrp_UNet3D_filter_sweep_0002_visualizer_aux(config_data,model_label_names_collection, 'fraction_clamp_filter', color_list, verbose=0)
	plt.show()