from utils.utils import * 
from dataio.dataISLES2017 import ISLES2017mass
import utils.vis as vi

def test(config_dir):
	config_raw_data = json_to_dict(config_dir)
	config_data = prepare_config(config_raw_data)
	if config_data['debug_test_mode'] == 'test_load_ISLES2017': test_load_ISLES2017(config_dir)
	elif config_data['debug_test_mode'] == 'test_load_ISLES2017b': test_load_ISLES2017b(config_dir)
	elif config_data['debug_test_mode'] == 'test_load_many_ISLES2017': test_load_many_ISLES2017(config_dir)
	elif config_data['debug_test_mode'] == 'test_data_augmentation': test_data_augmentation(config_dir)
	else: raise Exception('No valid mode chosen.')

'''
Available testing modes
'''
def test_data_augmentation(config_dir):
	print("test_data_augmentation")
	config_raw_data = json_to_dict(config_dir)
	config_data = prepare_config(config_raw_data)
	
	case_number = 1
	case_type = 'training'
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
	'''
	one_case['imgobj']['ADC'].shape: (192, 192, 19)
	one_case['imgobj']['MTT'].shape: (192, 192, 19)
	one_case['imgobj']['rCBF'].shape: (192, 192, 19)
	one_case['imgobj']['rCBV'].shape: (192, 192, 19)
	one_case['imgobj']['Tmax'].shape: (192, 192, 19)
	one_case['imgobj']['TTP'].shape: (192, 192, 19)
	one_case['imgobj']['OT'].shape: (192, 192, 19)
	one_case['case_number']: 4
	type(one_case['header']): <class 'nibabel.nifti1.Nifti1Header'>
	one_case['affine'].shape: (4, 4)
	one_case['MRS']: 4
	one_case['ttMRS']: 90
	one_case['TICI']: 3
	one_case['TSS']: 116
	one_case['TTT']: 93
	one_case['smir_id']: 127206
	'''
	one_case_modified = one_case

	vis = vi.SlidingVisualizer()
	vis.vis2(one_case,one_case_modified)
	plt.show()


def test_load_ISLES2017(config_dir):
	print("test_load_ISLES2017()")
	config_raw_data = json_to_dict(config_dir)
	config_data = prepare_config(config_raw_data)
	case_numbers = range(1,49)

	ISLESDATA = ISLES2017mass()
	ISLESDATA.verbose = 20
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	case_type = 'training'
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	for case_number in case_numbers:
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
	'''
		c_no|smir_id| ADC              | MTT              | rCBF             | rCBV             | Tmax             | TTP              | OT               |
		1   |127014 | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   |
		2   |127094 | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   |
		(!) D:/Desktop@D/meim2venv/meim3/data/isles2017\training_3 does not exist. Ignoring.
		4   |127206 | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   |
		5   |127214 | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   |
		6   |127222 | (256, 256, 24)   | (256, 256, 24)   | (256, 256, 24)   | (256, 256, 24)   | (256, 256, 24)   | (256, 256, 24)   | (256, 256, 24)   |
		7   |127230 | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   |
		...
		47  |188994 | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   | (128, 128, 25)   |
		48  |189002 | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   | (192, 192, 19)   |
		----------------------------------------------------------------------------------------------------------------------------------------------------------------
		min |       | 0                | -2.288327        | -455.3503        | -58.81981        | -1.993951        | -1.2876743       | 0                |
		max |       | 4095             | 89.48501         | 207.20389        | 40.0             | 123.06382        | 121.76149        | 1                |
		mean|       | 195.55737        | 1.07344          | 3.85694          | 0.37295          | 1.37612          | 6.11271          | 0.00549          |
	
	From the result, let us make the following naive normalization specifications. We convert to [0,1] from the following interval.
	 | ADC              0, 5000
	 | MTT              -5, 100
	 | rCBF             -500, 300
	 | rCBV             -60, 40
	 | Tmax             -5, 150
	 | TTP              -2, 150
	 | OT               NONE
	'''
	print("\nPrinting data shapes.")
	print("  %-4s|%-7s| %-16s | %-16s | %-16s | %-16s | %-16s | %-16s | %-16s |"%('c_no', 'smir_id','ADC','MTT' , 'rCBF', 'rCBV', 'Tmax','TTP','OT'))

	this_stats = {}
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	for modality_label in canonical_modalities_label: this_stats[modality_label] = {'shape': 0, 'min': np.inf, 'max': -np.inf, 'mean':[]}
	for case_number in case_numbers:
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
		if one_case is None: continue
		ADC,MTT,rCBF,rCBV,Tmax,TTP,OT = 0,0,0,0,0,0,0
		for modality_label in canonical_modalities_label:
			if modality_label in one_case['imgobj']: 
				this_stats[modality_label]['shape'] = one_case['imgobj'][modality_label].shape
				if this_stats[modality_label]['min'] > np.min(one_case['imgobj'][modality_label]): this_stats[modality_label]['min'] = np.min(one_case['imgobj'][modality_label])
				if this_stats[modality_label]['max'] < np.max(one_case['imgobj'][modality_label]): this_stats[modality_label]['max'] = np.max(one_case['imgobj'][modality_label])
				this_stats[modality_label]['mean'].append(np.mean(one_case['imgobj'][modality_label]))
		print("  %-4s|%-7s| %-16s | %-16s | %-16s | %-16s | %-16s | %-16s | %-16s |"%(str(one_case['case_number']), str(one_case['smir_id']), 
			this_stats['ADC']['shape'], this_stats['MTT']['shape'], this_stats['rCBF']['shape'], 
			this_stats['rCBV']['shape'], this_stats['Tmax']['shape'], this_stats['TTP']['shape'],
			this_stats['OT']['shape'] ))
	for modality_label in canonical_modalities_label: 
		if len(this_stats[modality_label]['mean']) > 0: this_stats[modality_label]['mean'] = round(np.mean(this_stats[modality_label]['mean']),5)
		else: this_stats[modality_label]['mean'] = "NA" 
	print("-"*160)
	print("  %-4s|%-7s| %-16s | %-16s | %-16s | %-16s | %-16s | %-16s | %-16s |"%('min', '', this_stats['ADC']['min'], this_stats['MTT']['min'], this_stats['rCBF']['min'], 
			this_stats['rCBV']['min'], this_stats['Tmax']['min'], this_stats['TTP']['min'], this_stats['OT']['min']))
	print("  %-4s|%-7s| %-16s | %-16s | %-16s | %-16s | %-16s | %-16s | %-16s |"%('max', '', this_stats['ADC']['max'], this_stats['MTT']['max'], this_stats['rCBF']['max'], 
			this_stats['rCBV']['max'], this_stats['Tmax']['max'], this_stats['TTP']['max'], this_stats['OT']['max']))
	print("  %-4s|%-7s| %-16s | %-16s | %-16s | %-16s | %-16s | %-16s | %-16s |"%('mean', '', this_stats['ADC']['mean'], this_stats['MTT']['mean'], this_stats['rCBF']['mean'], 
			this_stats['rCBV']['mean'], this_stats['Tmax']['mean'], this_stats['TTP']['mean'], this_stats['OT']['mean']))

def test_load_ISLES2017b(config_dir):
	'''
	one_case['imgobj']['ADC'].shape: (192, 192, 19)
	one_case['imgobj']['MTT'].shape: (192, 192, 19)
	one_case['imgobj']['rCBF'].shape: (192, 192, 19)
	one_case['imgobj']['rCBV'].shape: (192, 192, 19)
	one_case['imgobj']['Tmax'].shape: (192, 192, 19)
	one_case['imgobj']['TTP'].shape: (192, 192, 19)
	one_case['imgobj']['OT'].shape: (192, 192, 19)
	one_case['case_number']: 4
	type(one_case['header']): <class 'nibabel.nifti1.Nifti1Header'>
	one_case['affine'].shape: (4, 4)
	one_case['MRS']: 4
	one_case['ttMRS']: 90
	one_case['TICI']: 3
	one_case['TSS']: 116
	one_case['TTT']: 93
	one_case['smir_id']: 127206
'''
	print("test_load_ISLES2017b()")
	config_raw_data = json_to_dict(config_dir)
	config_data = prepare_config(config_raw_data)

	ISLESDATA = ISLES2017mass()
	ISLESDATA.verbose = 20
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	case_type = 'training'
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	canonical_modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP',6:'OT'}

	z_index = 0
	modality_index = 0
	case_number = 1
	one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
	vis = vi.SlidingVisualizer()
	vis.vis1(one_case)
	plt.show()

	
def test_load_many_ISLES2017(config_dir):
	print("test_load_many_ISLES2017")
	config_raw_data = json_to_dict(config_dir)
	config_data = prepare_config(config_raw_data)
	config_data['batch_size'] = 4
	printing_config(config_data)
	
	case_numbers = range(1,49)
	case_type = 'training'
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0001(case_type, case_numbers, config_data,normalize=True)
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=1, batch_size=config_data['batch_size'], shuffle=True)
	print("  trainloader loaded!")