from utils.utils import *
# from dataio.dataISLES2017 import ISLES2017mass

def visualize_evaluation_for_test_submission(config_data):
	print('visualize_evaluation_for_test_submission()')

	SERIES_NAME = 'UNet3D_XDGX22'
	# SERIES_NAME = 'UNet3D_SDGX21'
	output_folder_name = 'output'
	# dataname = 'SMIR.etjoa001_UNet3D_test_1.129319.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_2.129382.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_3.129410.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_4.129417.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_5.129424.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_6.129431.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_7.129438.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_19.129375.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_29.212307.nii'
	# dataname = 'SMIR.etjoa001_UNet3D_test_30.212315.nii'
	# dataname =  'SMIR.etjoa001_UNet3D_test_32.212331.nii'
	dataname = 'SMIR.etjoa001_UNet3D_test_33.212339.nii'
	# dataname = 'SMIR.etjoa003_UNet3D_test_11.129333.nii'
	
	working_dir = config_data['working_dir']
	data_path = os.path.join(working_dir,'checkpoints', SERIES_NAME, output_folder_name, dataname)
	img = nib.load(data_path)
	imgobj = np.array(img.dataobj)

	print('imgobj.shape:%s'%(str(imgobj.shape)))
	print(sum(imgobj.reshape(-1)))