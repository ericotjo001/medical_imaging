import os, pydicom, argparse
from dicom_handler import *
from os import listdir
# from pydicom.data import get_testdata_files
"""
D:
cd Desktop@D/dicom_conv
Scripts\activate
python ctpan.py -h
"""
def main():
	if arg_dict['mode'] == 'stack_all':
		stack_all(arg_dict['config_dir'])
		return
	if arg_dict['mode'] == 'generate_config':
		generate_config(arg_dict['dir'])
		return
	if arg_dict['mode'] == 'mass_observe_labels':
		all_patients_folder_dir = arg_dict['dir']
		mass_observe_labels(all_patients_folder_dir)
		return
	this_dir = os.getcwd()
	temp = listdir(arg_dict['dir'])[0]
	temp1 = listdir(os.path.join(arg_dict['dir'], temp))[0]
	one_patient_folder_dir = os.path.join(arg_dict['dir'], temp, temp1)
	if arg_dict['mode'] == 'verify_info':
		verify_info(one_patient_folder_dir, print_info=0)
	if arg_dict['mode'] == 'slice_info':
		slice_number = arg_dict['tag_number']
		verify_info(one_patient_folder_dir, print_only=int(slice_number), print_info=0)
	if arg_dict['mode'] == 'another_view':	
		anotherview(one_patient_folder_dir, print_info=0)
	if arg_dict['mode'] == 'z_position':
		extract_position(one_patient_folder_dir, print_info=0)
	if arg_dict['mode'] == 'stack_one':
		stack_one(one_patient_folder_dir, print_info=200)
	if arg_dict['mode'] == 'stack_one_and_save':
		config_dir = None
		if arg_dict['config_dir'] is not None:
			config_dir = arg_dict['config_dir']
		stack_one(one_patient_folder_dir, save_folder=arg_dict['target_folder'], 
			save_name=arg_dict['out_name'], config_dir=config_dir,
			print_info=100)
	if arg_dict['mode'] == 'mass_observe':
		all_patients_folder_dir = arg_dict['dir']
		mass_observe(all_patients_folder_dir, print_info=100)
	

	
if __name__=='__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
		description=DESCRIPTION)
	parser.add_argument('--mode')
	parser.add_argument('--dir')
	parser.add_argument('--tag_number')
	parser.add_argument('--target_folder')
	parser.add_argument('--out_name')
	parser.add_argument('--config_dir')
	args = parser.parse_args()
	arg_dict = {
		'mode': args.mode,
		'dir': args.dir,
		'tag_number': args.tag_number,
		'target_folder': args.target_folder,
		'out_name': args.out_name,
		'config_dir': args.config_dir
	}
	main()
