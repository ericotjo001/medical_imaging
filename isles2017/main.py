from utils.utils import * 
from pipeline.entry import print_info
from pipeline.entry import create_config_file
from pipeline.entry import train
from pipeline.entry import evaluation
from pipeline.entry import lrp
from pipeline.entry import shortcut_sequence
from pipeline.entry import visual
import test_cases.test as te

'''
For convenience:
D:
cd Desktop@D/meim2venv
Scripts\activate
cd meim3
python main.py

Important information to run this script can be found in utils/utils.py and the modules it imports from
'''
def main():
	if arg_dict['config_dir'] is None: config_dir = 'config.json'
	else: config_dir = arg_dict['config_dir']

	if arg_dict['mode'] == 'create_config_file': 
		create_config_file(config_dir); return
	elif arg_dict['mode'] == 'test': 
		te.test(config_dir); return
	
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)
	config_data['console_submode'] = arg_dict['submode']
	config_data['console_subsubmode'] = arg_dict['subsubmode']
	config_data['console_case_number'] = arg_dict['case_number']

	if arg_dict['mode'] is None or arg_dict['mode']=='info': 
		print_info(config_data)
	elif arg_dict['mode'] == 'more_info_on_publication_1':
		print(description_on_publication_1)
	elif arg_dict['mode'] == 'more_info_on_publication_2':
		print(description_on_publication_2)
	elif arg_dict['mode'] == 'train': 
		train(config_data)
	elif arg_dict['mode'] == 'evaluation': 
		evaluation(config_data)
	elif arg_dict['mode'] == 'lrp': 
		lrp(config_data)
	elif arg_dict['mode'] == 'visual': 
		visual(config_data)
	elif arg_dict['mode'] == 'shortcut_sequence': 
		shortcut_sequence(config_data, mode=arg_dict['shortcut_mode'])
	elif arg_dict['mode'] == 'results_publication_1':
		from pipeline.get_results.entry import select_mode_for_getting_results
		select_mode_for_getting_results(config_data)
	elif arg_dict['mode'] == 'results_publication_2':
		from pipeline.get_results.entry import select_mode_for_getting_results_2
		select_mode_for_getting_results_2(config_data)
	else:
		print('invalid mode!')
		print_info(config_data)

if __name__=='__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
		description=DESCRIPTION)
	parser.add_argument('--mode', help='mode. See utils/utils.py')
	parser.add_argument('--submode')
	parser.add_argument('--subsubmode')
	parser.add_argument('--config_dir', help='.json configuration file directory')
	parser.add_argument('--case_number',help='case number, for general purpose')
	parser.add_argument('--shortcut_mode',help='shortcut_mode, see entry.py shortcut_sequence()')
	args = parser.parse_args()
	arg_dict = {
		'mode': args.mode,
		'config_dir': args.config_dir,
		'shortcut_mode': args.shortcut_mode,
		'case_number': args.case_number,
		'submode': args.submode,
		'subsubmode': args.subsubmode
	}
	main()

