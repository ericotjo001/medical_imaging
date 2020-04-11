from utils.utils import * 

def select_mode_for_getting_results(config_data):
	print('select_mode_for_getting_results()')
	if config_data['console_submode'] is None:
		print(description_on_publication_1)
	elif config_data['console_submode']=='example_process_get_main_figure':
		from pipeline.get_results.results_publication_1.main_result import example_process_get_main_figure
		example_process_get_main_figure(config_data)
	elif config_data['console_submode']== 'example_process_get_filters_comparison_figure':
		from pipeline.get_results.results_publication_1.result_filters_comparison import example_process_get_filters_comparison_figure
		example_process_get_filters_comparison_figure(config_data)
	else:
		print(description_on_publication_1)



def select_mode_for_getting_results_2(config_data):
	print('select_mode_for_getting_results_2()')
	if config_data['console_submode'] is None:
		print(description_on_publication_2)
	else:
		print(description_on_publication_2)