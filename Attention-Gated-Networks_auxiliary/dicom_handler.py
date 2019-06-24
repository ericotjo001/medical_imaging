from utils import *

def stack_all(config_dir):
	assert(os.path.exists(config_dir))	
	config = json_to_dict(config_dir)
	main_dir = config['directory']['multiple_patients']
	suffix_check = config['directory']['suffix_check']
	save_folder = config['directory']['save_folder']
	print("main_dir\n",main_dir)
	print("suffix_check:", suffix_check)
	for patientID in listdir(main_dir):
		if patientID[:len(suffix_check)] == suffix_check:
			print(patientID)
			temp0 = os.path.join(main_dir, patientID)
			temp = listdir(temp0)[0]
			temp1 = listdir(os.path.join(temp0, temp))[0]
			one_patient_folder_dir = os.path.join(temp0, temp, temp1)
			stack_one(one_patient_folder_dir, save_folder=save_folder, save_name=patientID, config_dir=config_dir, print_info=0)
		
	return

def mass_observe_labels(all_patients_folder_dir, print_info=100):
	if print_info > 99: print("  %-16s | %-16s "%("patient ID", "np.shape"))
		"""
		Example output:
		patient ID       | np.shape
		label0001.nii.gz | (512, 512, 240)
		label0002.nii.gz | (512, 512, 195)
		label0003.nii.gz | (512, 512, 216)
		label0004.nii.gz | (512, 512, 221)
		label0005.nii.gz | (512, 512, 210)
		label0006.nii.gz | (512, 512, 223)
		label0007.nii.gz | (512, 512, 201)
		label0008.nii.gz | (512, 512, 205)
		label0009.nii.gz | (512, 512, 196)
		...
		label0081.nii.gz | (512, 512, 209)
		label0082.nii.gz | (512, 512, 226)
		"""
	try:
		import nibabel as nib
	except:
		raise Exception('Requires nibabel. Install using pip install nibabel.')
	for patientID in listdir(all_patients_folder_dir):
		full_path = os.path.join(all_patients_folder_dir, patientID)
		img = nib.load(full_path)
		print("  %-16s | %-16s"%(patientID, str(np.shape(img))))
		
	return

def mass_observe(all_patients_folder_dir, print_info=100):
	"""
	Data source: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
	The default downloaded data is saved in directory format like:
	  PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/000000.dcm

	Input:
	  all_patients_folder_dir is the path to folder which contains each patient's file
	    all_patients_folder_dir/PANCREAS001
	    all_patients_folder_dir/PANCREAS002
	    ...

	Example printed output:
		patient ID       | this_min         | this_max
		PANCREAS_0001    | -1024.0          | 2421.0
		PANCREAS_0002    | -1024.0          | 1253.0
		PANCREAS_0003    | -1024.0          | 1410.0
		PANCREAS_0004    | -1024.0          | 1784.0
		PANCREAS_0005    | -2048.0          | 1371.0
		PANCREAS_0007    | -1024.0          | 1432.0
		PANCREAS_0008    | -2048.0          | 1538.0
		PANCREAS_0009    | -2048.0          | 1500.0
		PANCREAS_0010    | -1024.0          | 1421.0
		PANCREAS_0011    | -1024.0          | 1310.0
		PANCREAS_0012    | -1024.0          | 3071.0
		Overall min, max = (-1024.000000, 3071.000000)
	"""
	this_dir = os.getcwd()
	overall_min, overall_max = np.inf, -np.inf
	if print_info > 99: print("  %-16s | %-16s | %-16s"%("patient ID", "this_min", "this_max"))
	for patientID in listdir(all_patients_folder_dir):
		temp0 = os.path.join(all_patients_folder_dir, patientID)
		temp = listdir(temp0)[0]
		temp1 = listdir(os.path.join(temp0, temp))[0]
		one_patient_folder_dir = os.path.join(temp0, temp, temp1)
		# print(patient_dir)
		_, this_min, this_max = stack_one(one_patient_folder_dir, print_info=0)
		if print_info > 99: print("  %-16s | %-16s | %-16s"%(patientID, str(this_min), str(this_max)))
		if overall_min < this_min: overall_min = this_min
		if overall_max > this_max: overall_max = this_max
	print("  Overall min, max = (%f, %f)"%(this_min, this_max))
	return

def stack_one(one_patient_folder_dir, save_folder=None, save_name=None,
	config_dir=None, print_info=0):
	"""
	Data source: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
	The default downloaded data is saved in directory format like:
	  PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/000000.dcm
	
	Input:
	  one_patient_folder_dir is the path/to/PANCREAS_0001
	
	"""
	z_pos = []
	slice_list = listdir(one_patient_folder_dir)
	slice_shape = None
	for i, slice_name in enumerate(slice_list):
		data_path = os.path.join(one_patient_folder_dir, slice_name)
		ds = pydicom.dcmread(data_path)

		if slice_shape is None: slice_shape =  ds.pixel_array.shape
		else: assert(np.all(slice_shape == ds.pixel_array.shape)) 
		for j, dss in enumerate(ds):
			if j == 31: 
				if print_info > 199: print("  ",j , dss.value, end=' ')
				if print_info > 199: print("  slice_shape: ", slice_shape)
				z_pos.append(int(dss.value[2]))

	cube = np.zeros(shape=(slice_shape[0], slice_shape[1], len(slice_list)))

	for i, slice_name in enumerate(slice_list):
		data_path = os.path.join(one_patient_folder_dir, slice_name)
		ds = pydicom.dcmread(data_path)
		# slice_array = ds.pixel_array
		cube[:,:,z_pos[i]] = ds.pixel_array	
	if print_info > 99: print("  cube.shape:", cube.shape)

	if config_dir is not None:
		config = json_to_dict(config_dir)
		do_normalize = bool(config['normalize'][0])
		ntemp = config['normalize'][1]
		global_max, global_min, target_max, target_min = float(ntemp['global_max']), float(ntemp['global_min']), float(ntemp['target_max']), float(ntemp["target_min"])
		if do_normalize:
			cube = normalize_numpy_array(cube, minx=target_min, maxx=target_max, minx0=global_min, maxx0=global_max)

	if save_folder is not None and save_name is not None:
		data_path = os.path.join(one_patient_folder_dir, slice_list[0])
		ds = pydicom.dcmread(data_path)
		ds0 = copy.copy(ds)
		ds0.PixelData = cube.tobytes()
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		if print_info > 19:
			this_min, this_max = np.min(cube), np.max(cube)
			print("  np.min(cube) = %f, np.max(cube) = %f"%(this_min, this_max))
			print("  stack_one(). save_folder:",save_folder, end='')
			print("  save_name  :", save_name)
		ds0.save_as(os.path.join(save_folder, save_name))
	else:
		this_min, this_max = np.min(cube),np.max(cube)
		if print_info > 19: print("  np.min(cube) = %f, np.max(cube) = %f"%(this_min, this_max))
		return cube, this_min, this_max
	return

def extract_position(one_patient_folder_dir, print_info=0):
	"""
	Data source: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
	The default downloaded data is saved in directory format like:
	  PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/000000.dcm
	
	Input:
	  one_patient_folder_dir is the path/to/PANCREAS_0001

	If the bool printed is true, then it means that the folder contains a series of dcm files, 000000.dcm, 000001.dcm, ...
	  such that the z positions form an integer sequence of increment 1.
	
	Example output:
	Check: bool=1 compared to range(-239,1)
  		When unsorted, bool=0
	"""
	z_pos = []

	for i, slice_name in enumerate(listdir(one_patient_folder_dir)):
		data_path = os.path.join(one_patient_folder_dir, slice_name)
		ds = pydicom.dcmread(data_path)
		for j, dss in enumerate(ds):
			if j==31: 
				# print("  ",j , dss.value)
				z_pos.append(int(dss.value[2]))
	x_min, x_max = int(np.min(z_pos)), int(np.max(z_pos))
	temp = np.sort(z_pos)
	if print_info>199:
		for z in temp:
			print(z, end=',')
	check =np.all(temp==range(x_min, x_max+1))
	print("\nCheck: bool=%d compared to range(%d,%d)"%(int(check),x_min, x_max+1 ))
	print("  When unsorted, bool=%d"%(z_pos==range(x_min, x_max+1)))
	return

def verify_info(one_patient_folder_dir, print_only=None, print_info=0):
	"""
	COnsider dataset downloaded from "https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT"
	The default downloaded data is saved in directory format like:
	  PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/000000.dcm
	Input:
	  one_patient_folder_dir is the path/to/PANCREAS_0001

	For a patient, say PANCREAS_0001, the data consists of 
	  1. 000000.dcm
	  2. 000001.dcm
	  3. ...
	Each is a 2D slice. What this function shows it that some tags are indeed different, for example "Instance Number"
	 though the difference might not be important
	"""
	for i, slice_name in enumerate(listdir(one_patient_folder_dir)):
		"""
		each ds is like this:
		0 (0008, 0005) Specific Character Set              CS: 'ISO_IR 100'
		1 (0008, 0008) Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'AXIAL']
		2 (0008, 0016) SOP Class UID                       UI: CT Image Storage
		3 (0008, 0018) SOP Instance UID                    UI: 1.2.826.0.1.3680043.2.1125.1.23216003808436901505928834695350977
		4 (0008, 0020) Study Date                          DA: '20151124'
		5 (0008, 0030) Study Time                          TM: '165447.754086'
		6 (0008, 0050) Accession Number                    SH: ''
		7 (0008, 0060) Modality                            CS: 'CT'
		8 (0008, 0064) Conversion Type                     CS: 'DV'
		9 (0008, 0070) Manufacturer                        LO: ''
		10 (0008, 0080) Institution Name                    LO: 'NIH'
		11 (0008, 0090) Referring Physician's Name          PN: ''
		12 (0008, 1030) Study Description                   LO: 'Pancreas'
		13 (0008, 103e) Series Description                  LO: 'Pancreas'
		14 (0010, 0010) Patient's Name                      PN: 'PANCREAS_0001'
		15 (0010, 0020) Patient ID                          LO: 'PANCREAS_0001'
		16 (0010, 0030) Patient's Birth Date                DA: ''
		17 (0010, 0040) Patient's Sex                       CS: ''
		18 (0013, 0010) Private Creator                     LO: 'CTP'
		19 (0013, 1010) Private tag data                    UN: b'Pancreas-CT '
		20 (0013, 1013) Private tag data                    UN: b'93781505'
		21 (0018, 0010) Contrast/Bolus Agent                LO: ''
		22 (0018, 0015) Body Part Examined                  CS: 'PANCREAS'
		23 (0018, 0050) Slice Thickness                     DS: ''
		24 (0018, 0060) KVP                                 DS: ''
		25 (0020, 000d) Study Instance UID                  UI: 1.2.826.0.1.3680043.2.1125.1.38381854871216336385978062044218957
		26 (0020, 000e) Series Instance UID                 UI: 1.2.826.0.1.3680043.2.1125.1.68878959984837726447916707551399667
		27 (0020, 0010) Study ID                            SH: 'PANCREAS_0001'
		28 (0020, 0011) Series Number                       IS: ''
		29 (0020, 0012) Acquisition Number                  IS: ''
		30 (0020, 0013) Instance Number                     IS: "106"
		31 (0020, 0032) Image Position (Patient)            DS: ['0', '0', '-105']
		32 (0020, 0037) Image Orientation (Patient)         DS: ['1', '0', '0', '0', '-1', '0']
		33 (0020, 0052) Frame of Reference UID              UI: 1.2.826.0.1.3680043.2.1125.1.45138396560156236976616409747397611
		34 (0020, 1040) Position Reference Indicator        LO: ''
		35 (0028, 0002) Samples per Pixel                   US: 1
		36 (0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
		37 (0028, 0010) Rows                                US: 512
		38 (0028, 0011) Columns                             US: 512
		39 (0028, 0030) Pixel Spacing                       DS: ['0.859375', '0.859375']
		40 (0028, 0100) Bits Allocated                      US: 16
		41 (0028, 0101) Bits Stored                         US: 16
		42 (0028, 0102) High Bit                            US: 15
		43 (0028, 0103) Pixel Representation                US: 1
		44 (0028, 1052) Rescale Intercept                   DS: "0"
		45 (0028, 1053) Rescale Slope                       DS: "1"
		46 (7fe0, 0010) Pixel Data                          OW: Array of 524288 bytes
		"""
		data_path = os.path.join(one_patient_folder_dir, slice_name)
		ds = pydicom.dcmread(data_path)
		print(slice_name) # e.g. "000000.dcm"		
		# print(type(ds))	# each ds is a <class 'pydicom.dataset.FileDataset'>
		for j, dss in enumerate(ds):
			if print_only is None:
				print("  ",j , dss) # each dss is a <class 'pydicom.dataelem.DataElement'>
			else:
				assert(0 <= print_only and print_only <= 46)
				if j==print_only:print("  ",j , dss)
	return

def anotherview(one_patient_folder_dir, print_info=0):
	"""
	COnsider dataset downloaded from "https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT"
	The default downloaded data is saved in directory format like:
	  PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/000000.dcm
	Input:
	  one_patient_folder_dir is the path/to/PANCREAS_0001

	not equal at n = 3 (0008, 0018) SOP Instance UID
	not equal at n = 5 (0008, 0030) Study Time
	not equal at n = 30 (0020, 0013) Instance Number
	not equal at n = 31 (0020, 0032) Image Position (Patient)
	"""
	transposed_list = []
	tag_names = []
	for i in range(46): transposed_list.append([])
	for i, slice_name in enumerate(listdir(one_patient_folder_dir)):
		data_path = os.path.join(one_patient_folder_dir, slice_name)
		ds = pydicom.dcmread(data_path)
		dsiter = ds.__iter__()
		for n, j in enumerate(dsiter):
			# print(" %2d %-20s| %-20s| %-4s | %-4s | %-20s"%(n, str(j.name), str(j.tag), str(j.VR), str(j.VM), str(j.value)))
			"""
			0 Specific Character Set| (0008, 0005)        | CS   | 1    | ISO_IR 100
			1 Image Type          | (0008, 0008)        | CS   | 3    | ['ORIGINAL', 'PRIMARY', 'AXIAL']
			2 SOP Class UID       | (0008, 0016)        | UI   | 1    | 1.2.840.10008.5.1.4.1.1.2
			3 SOP Instance UID    | (0008, 0018)        | UI   | 1    | 1.2.826.0.1.3680043.2.1125.1.50678515635029708854661081345442613
			4 Study Date          | (0008, 0020)        | DA   | 1    | 20151124
			5 Study Time          | (0008, 0030)        | TM   | 1    | 165447.769002
			6 Accession Number    | (0008, 0050)        | SH   | 1    |
			7 Modality            | (0008, 0060)        | CS   | 1    | CT
			8 Conversion Type     | (0008, 0064)        | CS   | 1    | DV
			9 Manufacturer        | (0008, 0070)        | LO   | 1    |
			10 Institution Name    | (0008, 0080)        | LO   | 1    | NIH
			11 Referring Physician's Name| (0008, 0090)        | PN   | 1    |
			12 Study Description   | (0008, 1030)        | LO   | 1    | Pancreas
			13 Series Description  | (0008, 103e)        | LO   | 1    | Pancreas
			14 Patient's Name      | (0010, 0010)        | PN   | 1    | PANCREAS_0001
			15 Patient ID          | (0010, 0020)        | LO   | 1    | PANCREAS_0001
			16 Patient's Birth Date| (0010, 0030)        | DA   | 1    |
			17 Patient's Sex       | (0010, 0040)        | CS   | 1    |
			18 Private Creator     | (0013, 0010)        | LO   | 1    | CTP
			19 Private tag data    | (0013, 1010)        | UN   | 1    | b'Pancreas-CT '
			20 Private tag data    | (0013, 1013)        | UN   | 1    | b'93781505'
			21 Contrast/Bolus Agent| (0018, 0010)        | LO   | 1    |
			22 Body Part Examined  | (0018, 0015)        | CS   | 1    | PANCREAS
			23 Slice Thickness     | (0018, 0050)        | DS   | 1    |
			24 KVP                 | (0018, 0060)        | DS   | 1    |
			25 Study Instance UID  | (0020, 000d)        | UI   | 1    | 1.2.826.0.1.3680043.2.1125.1.38381854871216336385978062044218957
			26 Series Instance UID | (0020, 000e)        | UI   | 1    | 1.2.826.0.1.3680043.2.1125.1.68878959984837726447916707551399667
			27 Study ID            | (0020, 0010)        | SH   | 1    | PANCREAS_0001
			28 Series Number       | (0020, 0011)        | IS   | 1    |
			29 Acquisition Number  | (0020, 0012)        | IS   | 1    |
			30 Instance Number     | (0020, 0013)        | IS   | 1    | 115
			31 Image Position (Patient)| (0020, 0032)        | DS   | 3    | ['0', '0', '-114']
			32 Image Orientation (Patient)| (0020, 0037)        | DS   | 6    | ['1', '0', '0', '0', '-1', '0']
			33 Frame of Reference UID| (0020, 0052)        | UI   | 1    | 1.2.826.0.1.3680043.2.1125.1.45138396560156236976616409747397611
			34 Position Reference Indicator| (0020, 1040)        | LO   | 1    |
			35 Samples per Pixel   | (0028, 0002)        | US   | 1    | 1
			36 Photometric Interpretation| (0028, 0004)        | CS   | 1    | MONOCHROME2
			37 Rows                | (0028, 0010)        | US   | 1    | 512
			38 Columns             | (0028, 0011)        | US   | 1    | 512
			39 Pixel Spacing       | (0028, 0030)        | DS   | 2    | ['0.859375', '0.859375']
			40 Bits Allocated      | (0028, 0100)        | US   | 1    | 16
			41 Bits Stored         | (0028, 0101)        | US   | 1    | 16
			42 High Bit            | (0028, 0102)        | US   | 1    | 15
			43 Pixel Representation| (0028, 0103)        | US   | 1    | 1
			44 Rescale Intercept   | (0028, 1052)        | DS   | 1    | 0
			45 Rescale Slope       | (0028, 1053)        | DS   | 1    | 1

			"""
			if i == 0: tag_names.append(str(j.tag))
			transposed_list[n].append([str(j.name), str(j.tag), str(j.VR), str(j.VM), str(j.value)])
			if n==45: break

	# for this_tag in tag_names:
	# 	print(this_tag)

	for n, g in enumerate(transposed_list):
		for i in range(len(g)-1):
			if not g[0] == g[i+1]: 
				print(" not equal at n = %d %s"%(n,tag_names[n]),g[0][0] )			 
				break
	return

def generate_config(config_dir):
	CONFIG_FILE = {
		'normalize' : [ "True",
			{
				'global_max': 10000,
				'global_min': -10000,
				'target_max': 1,
				'target_min': 0
			}	
		],
		'directory' : {
			'mutliple_patients': "D:/Desktop@D/Attention-Gated-Networks/data/TCIA_pancreas_in_DICOM_format/PANCREAS-CT-82",
			'save_folder': "D:/Desktop@D/dicom_conv/output_dir",
			'suffix_check': "PANCREAS"
		}
	}
	with open('config.json', 'w') as json_file:  
 	   json.dump(CONFIG_FILE, json_file, separators=(',', ': '), indent=2)
	with open('config_doc.txt', 'w') as txt:
		for one_line in CONFIG_ARG_EXPLANATIONS: txt.write(one_line)
	return

