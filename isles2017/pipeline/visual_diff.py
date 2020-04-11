from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass

def visual_diff_gen_0001(config_data, visual_config=None):
	print('visual_diff_gen_0001(). HARDCODED VARIABLES? YES')
	case_type = 'training'
	canonical_modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP',6:'OT'}

	if visual_config is None:
		visual_config = {
			'case_numbers': [16], # [1,4,8,],
			'DEPTH_INDEX' : 10,
			'NOW_SHOW' : 0,
			'CHANNEL_SELECTION' : 3,
			'defect_fraction':1.2			 
		}
	case_numbers = visual_config['case_numbers']
	DEPTH_INDEX = visual_config['DEPTH_INDEX']
	NOW_SHOW = visual_config['NOW_SHOW']
	CHANNEL_SELECTION = visual_config['CHANNEL_SELECTION']
	defect_fraction= visual_config['defect_fraction']


	config_data['dataloader']['resize'] = [192,192,19]
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	ISLESDATA.load_many_cases(case_type, case_numbers, config_data)

	isles_mri = torch.tensor([ISLESDATA.__getitem__(NOW_SHOW)[0]]).permute(0,1,4,3,2)
	isles_lesion = torch.tensor([ISLESDATA.__getitem__(NOW_SHOW)[1]]).permute(0,3,2,1).to(torch.float) 
	print('isles_mri.shape:',isles_mri.shape)
	print('isles_lesion.shape:',isles_lesion.shape)

	from dataio.data_diffgen import DG3D
	dg = DG3D(unit_size=(192,192), depth=19) # this size will be interpolated to 3D (19,192,192)
	_, x_unhealthy, y_lesion, _ = dg.generate_data_batches_in_torch(
		channel_size=6, batch_size=1,resize=config_data['dataloader']['resize'])
	print('x_unhealthy.shape:',x_unhealthy.shape )
	print('y_lesion.shape:',y_lesion.shape)

	isles_mri = isles_mri.cpu()
	isles_lesion = isles_lesion.cpu()
	x_unhealthy = x_unhealthy.cpu()
	y_lesion = y_lesion.cpu()

	mixed = isles_mri + defect_fraction*x_unhealthy
	mixed = torch.clamp(mixed,0,1)
	mixed_y = (y_lesion.to(torch.float)+isles_lesion.to(torch.float))>0
	print('mixed.shape:',mixed.shape)
	print('mixed_y.shape:',mixed_y.shape)

	isles_mri = isles_mri.permute(0,1,4,3,2).squeeze()
	isles_lesion = isles_lesion.permute(0,3,2,1).squeeze()

	x_unhealthy = x_unhealthy.permute(0,1,4,3,2).squeeze()
	y_lesion = y_lesion.permute(0,3,2,1).squeeze()

	mixed = mixed.permute(0,1,4,3,2).squeeze()
	mixed_y = mixed_y.permute(0,3,2,1).squeeze()


	cmap = 'inferno'
	fig = plt.figure()

	ax1 = fig.add_subplot(321)
	im1 = ax1.imshow(isles_mri[CHANNEL_SELECTION][:,:,DEPTH_INDEX],cmap=cmap)
	plt.colorbar(im1)

	ax2 = fig.add_subplot(322)
	ax2.imshow(isles_lesion[:,:,DEPTH_INDEX],cmap='Greys')
	
	ax3 = fig.add_subplot(323)
	im3 = ax3.imshow(x_unhealthy[CHANNEL_SELECTION][:,:,DEPTH_INDEX],cmap=cmap)
	plt.colorbar(im3)
	
	ax4 = fig.add_subplot(324)
	ax4.imshow(y_lesion[:,:,DEPTH_INDEX],cmap='Greys')
	
	ax5 = fig.add_subplot(325)
	im5 = ax5.imshow(mixed[CHANNEL_SELECTION][:,:,DEPTH_INDEX],cmap=cmap)
	plt.colorbar(im5)

	ax6 = fig.add_subplot(326)
	ax6.imshow(mixed_y[:,:,DEPTH_INDEX],cmap='Greys')

	this_title = '%s:%s'%(str(visual_config['case_numbers'][NOW_SHOW]),str(canonical_modalities_dict[CHANNEL_SELECTION]))
	ax1.set_title(this_title)
	
	plt.tight_layout()
	plt.show()