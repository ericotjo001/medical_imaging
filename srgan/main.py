from utils.utils import *
from pipeline.entry import train 
from pipeline.entry import evaluation
from pipeline.entry import shortcut_sequence
import tests.test as te


def main():
	if arg_dict['config_dir'] is None: config_dir = 'config.json'
	else: config_dir = arg_dict['config_dir']

	if arg_dict['mode'] is None or arg_dict['mode']=='info': print_info()
	elif arg_dict['mode'] == 'create_config_file': create_config_file(config_dir)
	elif arg_dict['mode'] == 'test': te.test(config_dir)
	elif arg_dict['mode'] == 'train': train(config_dir)
	elif arg_dict['mode'] == 'evaluation': evaluation(config_dir)
	# elif arg_dict['mode'] == 'lrp': lrp(config_dir)
	elif arg_dict['mode'] == 'shortcut_sequence': shortcut_sequence(config_dir)
	else: raise Exception('Invalid mode.')

if __name__=='__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
		description=DESCRIPTION)
	parser.add_argument('--mode', help='mode. See utils/utils.py')
	parser.add_argument('--config_dir', help='.json configuration file directory')
	parser.add_argument('--case_number',help='case number, for general purpose')
	args = parser.parse_args()
	arg_dict = {
		'mode': args.mode,
		'config_dir': args.config_dir
	}
	main()
