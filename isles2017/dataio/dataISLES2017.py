from utils.utils import *
from utils.custom_augment import Image3DAugment

DEBUG_load_many_cases_type0001 = 0
DEBUG_load_many_cases_type0002 = 0
DEBUG_load_type0003_aug = 0
DEBUG_AUG_NUMBER = 0 # default 0

'''
IMPT:
During data loading, data will not be transposed.
Transposition will occur during training.
Transposition is required because ConvNd from pytorch takes object in the shape (batch_size,C,D,H,W)
'''

class ISLES2017(data.Dataset):
	def __init__(self):
		super(ISLES2017, self).__init__()
		self.x = []	
		self.y = []
		self.data_size = None
		self.data_mode = None
		self.directory_path = None
		self.verbose=0

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.data_size

	def load_one_case(self, case_type, case_number, modalities_label):
		'''
		Assume self.directory_path is set.
		case_type = 'train' or 'test'
		case_number. See the number N from training_N or test_N in the isles folder.
		modalities_label is the list of modalities in the data that we want to extract.
		  For example, ['ADC','MTT' , 'rCBF', 'rCBV' , 'Tmax','TTP','OT']. Another modality is '4DPWI'
		'''
		if self.verbose > 99: print("\nISLES2017.load_one_case()")
		case_dir = os.path.join(self.directory_path, case_type + "_" + case_number)
		case_subdirs = self.__get_case_subdirs(case_dir, modalities_label)	
		if case_subdirs is None: print("(!) %s does not exist. Ignoring."%(case_dir)); return None
		if self.verbose > 199: self.__print_load_one_case(case_type, case_number, modalities_label, case_dir, case_subdirs)
		one_case = self.__load_one_case_type_1(case_type, case_number, case_subdirs)				
		return one_case
	
	def __get_case_subdirs(self, case_dir, modalities_label):
		case_subdirs = []
		if not os.path.exists(case_dir): return None
		for x in listdir(case_dir):
			for y in modalities_label:
				if x.find(y) >=0: case_subdirs.append([y,x]); break
		return case_subdirs

	def __print_load_one_case(self, case_type, case_number, modalities_label, case_dir, case_subdirs):
		print("  case_type=%s\n  case_number=%s"%(case_type, case_number))
		print("  modalities_label=%s"%(modalities_label))
		print("  case_dir: %s"%(case_dir))
		print("  case_subdir")
		for x in case_subdirs: print("    ",x)

	def __load_one_case_type_1(self, case_type, case_number, case_subdirs):
		if self.verbose > 99: print("  __load_one_case_type_1()")
		
		one_case = {'imgobj': {}}
		for modality_label, case_subdir in case_subdirs:
			data_path = os.path.join(self.directory_path, case_type + "_" + case_number ,case_subdir, case_subdir + '.nii')
			if self.verbose > 99: print("    %s [%s]"%(data_path, modality_label))
			img = nib.load(data_path)
			imgobj = np.array(img.dataobj)
			one_case['case_number'] = case_number
			one_case['imgobj'][modality_label] = imgobj
			if 'ADC' in modality_label: one_case = self.__get_header_and_affine(one_case, img)
			if 'MTT' in modality_label: one_case = self.__get_smir_id(one_case, case_subdir)
			self.__get_csv_data(one_case, case_type, case_number)
		if self.verbose > 249: self.__print_load_one_case_type_1(one_case)
		return one_case

	def __get_csv_data(self, one_case, case_type, case_number):
		# print("case_type:X%sX"%(case_type))
		if case_type == 'training': filename = 'ISLES2017_Training'

		out = read_csv(os.path.join(self.directory_path, filename))
		for i in range(len(out)):
			cast_number_csv=int(out[i][0][len(case_type)+1:])
			if cast_number_csv == int(case_number):
				MRS,ttMRS,TICI,TSS,TTT=out[i][1],out[i][2],out[i][3],out[i][4],out[i][5]		
				one_case['MRS'] = MRS
				one_case['ttMRS'] = ttMRS
				one_case['TICI'] = TICI
				one_case['TSS'] = TSS
				one_case['TTT'] = TTT
				break
		return 

	def __print_load_one_case_type_1(self, one_case):
		print("    __print_load_one_case_type_1()")
		'''
			one_case['imgobj']['ADC'].shape: (192, 192, 19)
			one_case['imgobj']['MTT'].shape: (192, 192, 19)
			one_case['imgobj']['rCBF'].shape: (192, 192, 19)
			one_case['imgobj']['rCBV'].shape: (192, 192, 19)
			one_case['imgobj']['Tmax'].shape: (192, 192, 19)
			one_case['imgobj']['TTP'].shape: (192, 192, 19)
			one_case['imgobj']['OT'].shape: (192, 192, 19)
			one_case['modality_label']: OT
			type(one_case['header']): <class 'nibabel.nifti1.Nifti1Header'>
			one_case['affine'].shape: (4, 4)
			one_case['smir_id']: 127014
		'''
		for xkey in one_case:
			if xkey == 'imgobj': 
				for modality_label in one_case['imgobj']: 
					print("      one_case['imgobj']['%s'].shape: %s"%(modality_label, str(one_case['imgobj'][modality_label].shape)))
			elif xkey == 'header': print("      type(one_case['header']): %s"%(type(one_case['header'])))
			elif xkey == 'affine': print("      one_case['affine'].shape: %s"%(str(one_case['affine'].shape)))
			else: print("      one_case['%s']: %s"%(str(xkey), str(one_case[xkey])))
		return

	def __get_header_and_affine(self, one_case, img):
		one_case['header'] = img.header
		one_case['affine'] = img.affine
		return one_case

	def __get_smir_id(self, one_case, case_subdir):
		pos1 = case_subdir.find("MR_MTT")
		checkpt1 = case_subdir[pos1:]
		smir_id = checkpt1[checkpt1.find(".")+1:]
		one_case['smir_id'] = smir_id
		return one_case

class ISLES2017mass(ISLES2017):
	def __init__(self):
		super(ISLES2017mass, self).__init__()
		
	def load_many_cases_type0001(self, case_type, case_numbers, config_data,
		normalize=True):
		'''
		type0001
		  x: load all 6 modalities into 1 array; 1 modality -> 1 channel. Every data size remains the same
		  y: segmentation groundtruth OT
		'''
		canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
		modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP'}
		data_size = 0
		for case_number in case_numbers:
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
			one_case = self.load_one_case(case_type, str(case_number), canonical_modalities_label)
			if one_case is None: continue
			data_size = data_size + 1
			s = one_case['imgobj']['ADC'].shape
			x1 = np.zeros(shape=(6,)+s)
			y = one_case['imgobj']['OT']

			for modality_key in modalities_dict:
				x1_component = one_case['imgobj'][modalities_dict[modality_key]]
				if normalize:  x1_component = normalize_numpy_array(x1_component,
					target_min=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][0],
					target_max=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][1],
					source_min=config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][0],
					source_max=config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][1], 
					verbose = 0)
				x1[modality_key,:,:,:] = x1_component
			self.x.append(x1)
			self.y.append(y)

			if DEBUG_load_many_cases_type0001: 
				print("\n *** DEBUG_load_many_cases_type0001:\n  %24s | %24s"%("np.max(x1[i])","np.min(x1[i])"))
				for i in range(6): print("  %24s | %24s"%(str(np.max(x1[i])),str(np.min(x1[i]))))
				break
		self.data_size = data_size		

	"""
	The normalization might not be good, because different data are stored in different order of intensities.
	
	def load_many_cases_type0002(self, case_type, case_numbers, config_data,
		normalize=True):		
		'''
		interp3d() resizes only x. y remains in the original shape. 
		'''
		canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
		modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP'}
		data_size = 0
		resize_shape = tuple(config_data['FCN_1']['resize'])

		for case_number in case_numbers:
			one_case = self.load_one_case(case_type, str(case_number), canonical_modalities_label)
			if one_case is None: continue
			data_size = data_size + 1
			s = one_case['imgobj']['ADC'].shape
			x1 = np.zeros(shape=(6,)+resize_shape)
			y = one_case['imgobj']['OT']

			for modality_key in modalities_dict:
				x1_component = one_case['imgobj'][modalities_dict[modality_key]]
				if normalize:  x1_component = normalize_numpy_array(x1_component,
					target_min=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][0],
					target_max=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][1],
					source_min=config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][0],
					source_max=config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][1], 
					verbose = 0)
				x1_component = torch.tensor(x1_component)
				x1[modality_key,:,:,:] = interp3d(x1_component,resize_shape)
			self.x.append(x1)
			self.y.append(y)

			if DEBUG_load_many_cases_type0002: 
				print("\n *** DEBUG_load_many_cases_type0001:\n  %24s | %24s"%("np.max(x1[i])","np.min(x1[i])"))
				for i in range(6): print("  %24s | %24s"%(str(np.max(x1[i])),str(np.min(x1[i]))))
				print("x1.shape=%s"%(str(x1.shape)))
				print("y.shape=%s"%(str(y.shape)))
				break
		self.data_size = data_size
	"""

	def load_many_cases_type0003(self, case_type, case_numbers, config_data,
		normalize=True):
		'''
		Normalization. In this data load type, we normalize against the min and max
		of each data point. Hence source_min_max configurations are irrelevant
		'''

		canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
		modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP'}
		data_size = 0
		resize_shape = tuple(config_data['dataloader']['resize'])

		available_case_numbers = []
		for case_number in case_numbers:
			one_case = self.load_one_case(case_type, str(case_number), canonical_modalities_label)
			if one_case is None: continue
			else: available_case_numbers.append(case_number); print("case_number:",case_number)
			s = one_case['imgobj']['ADC'].shape
			x1 = np.zeros(shape=(6,)+resize_shape)
			y = one_case['imgobj']['OT']

			for modality_key in modalities_dict:
				x1_component = one_case['imgobj'][modalities_dict[modality_key]]
				if normalize:  x1_component = normalize_numpy_array(x1_component,
					target_min=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][0],
					target_max=config_data['normalization'][modalities_dict[modality_key]+"_target_min_max"][1],
					source_min=np.min(x1_component),#config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][0],
					source_max=np.max(x1_component),#config_data['normalization'][modalities_dict[modality_key]+"_source_min_max"][1], 
					verbose = 0)
				x1_component = torch.tensor(x1_component)
				x1[modality_key,:,:,:] = interp3d(x1_component,resize_shape)

			y1 = torch.tensor(np.array(y,dtype=np.float))
			y1 = interp3d(y1,resize_shape).numpy()
			
			if config_data['augmentation']['type'] == 'no_augmentation':
				data_size = data_size + 1
				ytest = y1.reshape(-1)
				unique_list =[x for x in set(ytest)]
				assert(np.all(unique_list==[0.,1.]))
				self.x.append(x1)
				self.y.append(y1)		
			elif config_data['augmentation']['type'] == 'rotate_then_clip_translate':
				'''
				At this point x1, y are still numpy arrays
				x1 is C, w,h,d
				y is w,h,d
						'''
				assert(np.all(x1.shape[1:] == y1.shape))
				aug = Image3DAugment(x1[0].shape, verbose = 0)

				for i in range(config_data['augmentation']['number_of_data_augmented_per_case']):
					x2, y2 = self.load_type0003_aug001(i, x1, y1, aug)
					data_size = data_size + 1
					self.x.append(x2)
					self.y.append(y2)	
					if DEBUG_AUG_NUMBER > 0: 
						if i >=DEBUG_AUG_NUMBER: break
		self.data_size = data_size
		print("data size:%s"%(str(data_size)))
		print("case_numbers:%s"%(str(available_case_numbers)))

	def load_type0003_aug001(self, i, x1, y1, aug):
		x1s = x1.shape
		x2 = np.zeros(x1.shape)
		y2 = np.zeros(y1.shape)

		ytest = y1.reshape(-1)
		if DEBUG_load_type0003_aug:
			unique_list_before = [x for x in set(ytest)]
		if i > 0:
			aug_param = aug.generate_random_augment_params(verbose=0)
			
			for j in range(x1s[0]):
				x2[j,:,:,:] = aug.rotate_then_clip_translate(x1[j,:,:,:], aug_param)

			aug_param['crop'] = [0.,1.]
			y2 = aug.rotate_then_clip_translate(y1, aug_param)
			if DEBUG_load_type0003_aug:
				unique_list_mid = [x for x in set(y2.reshape(-1))]
			y2 = np.array(y2 > 0.5, dtype=np.float)
			
			if DEBUG_load_type0003_aug:
				ytest = y2.reshape(-1)
				unique_list =[x for x in set(ytest)]
				print("  unique_list_before = %s uniques_list_mid = %s unique_list = %s "%(str(unique_list_before), str(unique_list_mid),str(unique_list)))
		elif i==0:
			x2 = x1
			y2 = y1
			if DEBUG_load_type0003_aug:
				unique_list =[x for x in set(ytest)]
				print("  unique_list_before = %s unique_list = %s "%(str(unique_list_before),str(unique_list)))
		if DEBUG_load_type0003_aug:
			assert(np.all(unique_list==[0.,1.]) or np.all(unique_list==[0.]))
		return x2, y2