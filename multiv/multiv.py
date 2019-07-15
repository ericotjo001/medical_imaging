from multivtests import *

"""
D:
cd Desktop@D\meim2venv
Scripts\activate
cd multiv
python multiv.py
"""


def main():
	# print("hello")
	# print("dir path:      ",dir_path)
	# print("this folder:   ",this_folder)
	# print("parent_folder: ", parent_folder)
	# print("args.mode: ",args.mode)
	if args.mode is None:
		print(DESCRIPTION)
	elif arg_dict['mode'] == 'generate_config':
		this_dir = args.dir
		generate_config(this_dir=this_dir)
	elif arg_dict['mode'] == 'test':
		if arg_dict['submode'] == 'multiv_test1' or arg_dict['submode'] is None: 
			multiv_test1()
		if arg_dict['submode'] == 'multiv_uniform2D_test':
			multiv_uniform2D_test(config_dir=args.dir)
		if arg_dict['submode'] == 'multiv_uniform3D_test':
			multiv_uniform3D_test(config_dir=args.dir)
		if arg_dict['submode'] == 'multiv_unirand2D_test':
			multiv_unirand2D_test(config_dir=args.dir)
		if arg_dict['submode'] == 'multiv_unirand3D_test':
			multiv_unirand3D_test(config_dir=args.dir)
		if arg_dict['submode'] == 'dualview2D_test':
			dualview2D_test(config_dir=args.dir)
		if arg_dict['submode'] == 'dualview3D_test':
			dualview3D_test(config_dir=args.dir)
		if arg_dict['submode'] == 'dualuniform2D_test':
			dualuniform2D_test(config_dir=args.dir, verbose=10)
		if arg_dict['submode'] == 'dualuniform3D_test':
			dualuniform3D_test(config_dir=args.dir, verbose=10)

			
if __name__=='__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
		description=DESCRIPTION)
	parser.add_argument('--mode', help='Input mode.')
	parser.add_argument('--dir', help='Config file directory.')
	parser.add_argument('--submode', help='Input submode.')
	args = parser.parse_args()
	arg_dict = {
		'mode': args.mode,
		'dir': args.dir,
		'submode': args.submode
	}
	main()
