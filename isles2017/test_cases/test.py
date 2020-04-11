from utils.utils import * 
import utils.loss as uloss
from utils.evalobj import EvalObj
from dataio.dataISLES2017 import ISLES2017mass
import utils.vis as vi

def test(config_dir):
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)

	if config_data['debug_test_mode'] == 'test_load_ISLES2017': test_load_ISLES2017(config_data)
	elif config_data['debug_test_mode'] == 'test_load_many_ISLES2017': test_load_many_ISLES2017(config_data)
	elif config_data['debug_test_mode'] == 'test_data_augmentation': test_data_augmentation(config_data)
	elif config_data['debug_test_mode'] == 'test_save_one_for_submission': test_save_one_for_submission(config_data)
	else: raise Exception('No valid mode chosen.')

'''
Available testing modes
'''
def test_save_one_for_submission(config_data):
	print("test_save_one_for_submission()")
	ev = EvalObj()

	case_type = 'training'
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	ISLESDATA = ISLES2017mass()
	ISLESDATA.verbose = 20
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	
	# Note: at the point one_case is loaded, the shape is still (h,w,d)

	# # 1. 
	case_number = 1
	one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
	print("case_number:%s"%(str(case_number)))
	print("  one_case['imgobj']['OT'].shape:%s"%(str(one_case['imgobj']['OT'].shape)))
	for xkey in one_case: print(xkey) # just for observation
	original_shape = one_case['imgobj']['OT'].shape
	dummy_output_shape = (1,) + one_case['imgobj']['OT'].shape # assume at this point proper permutation has been done
	dummy_output = one_case['imgobj']['OT'].reshape(dummy_output_shape) + np.array(np.random.randint(0,1000,size=dummy_output_shape)>995).astype(np.float)
	dummy_output = torch.tensor(dummy_output)
	dummy_output_numpy = dummy_output.detach().cpu().numpy()
	y_ot_tensor = torch.tensor(one_case['imgobj']['OT'].reshape(dummy_output_shape))
	ISLESDATA.save_one_case(dummy_output_numpy.reshape(original_shape), one_case, case_type, case_number,config_data,desc='etjoa001_'+str(case_number))
	ev.save_one_case_evaluation(case_number, dummy_output, y_ot_tensor, config_data, dice=True)
	
	# # 2.
	case_number2 = 10
	one_case = ISLESDATA.load_one_case(case_type, str(case_number2), canonical_modalities_label)	
	print("case_number2:%s"%(str(case_number2)))
	print("  one_case['imgobj']['OT'].shape:%s"%(str(one_case['imgobj']['OT'].shape)))
	original_shape = one_case['imgobj']['OT'].shape
	dummy_output2_shape = (1,) + one_case['imgobj']['OT'].shape 
	dummy_output2 = one_case['imgobj']['OT']
	dummy_output2 = torch.tensor(dummy_output2.reshape(dummy_output2_shape))
	dummy_output2_numpy = dummy_output2.detach().cpu().numpy()
	ISLESDATA.save_one_case(dummy_output2_numpy.reshape(original_shape), one_case, case_type, case_number2,config_data, desc='etjoa001_'+str(case_number2))
	y_ot2_tensor = torch.tensor(one_case['imgobj']['OT'].reshape(dummy_output2_shape))
	ev.save_one_case_evaluation(case_number2, dummy_output2, y_ot2_tensor, config_data, dice=True)

def test_data_augmentation(config_data):
	print("test_data_augmentation")
	
	case_number = 1
	case_type = 'training'
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
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
	one_case_modified = {}
	one_case_modified['imgobj'] = {}
	for x_modality in canonical_modalities_label:
		if x_modality=='OT': continue
		x1_component = one_case['imgobj'][x_modality]
		x1_component = normalize_numpy_array(x1_component,target_min=0,target_max=1,source_min=np.min(x1_component),source_max=np.max(x1_component),)
		x1_component = np.clip(x1_component,0.0,0.5)
		one_case_modified['imgobj'][x_modality] = x1_component
	one_case_modified['imgobj']['OT'] = one_case['imgobj']['OT']

	vis = vi.SlidingVisualizer()
	vis.vis2(one_case,one_case_modified)
	plt.show()

def test_load_ISLES2017(config_data):
	print("test_load_ISLES2017()")
	case_numbers = range(1,49)

	ISLESDATA = ISLES2017mass()
	ISLESDATA.verbose = 20
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
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
	
	(!!!) Impt the following normalization appears to be suitable after observing the data. The intensities are not very compatible with each other
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

def test_load_many_ISLES2017(config_data):
	print("test_load_many_ISLES2017")
	config_data['batch_size'] = 4
	
	case_numbers = range(1,49)
	case_type = 'training'
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0003(case_type, case_numbers, config_data,normalize=True)
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=1, batch_size=config_data['batch_size'], shuffle=True)
	print("  trainloader loaded!")