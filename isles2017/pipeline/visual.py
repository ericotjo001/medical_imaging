from utils.utils import *
import utils.evalobj as ev
import pipeline.lrp as lr
import utils.vis as vi
from dataio.dataISLES2017 import ISLES2017mass

import pipeline.visual_header as vh

'''
Customize your visualization modes here
'''

def visual_select_submode(config_data):

	if config_data['visual_mode']=='plot_loss_0001': plot_loss_0001(config_data)
	elif config_data['visual_mode'] == 'lrp_UNet3D_overfit_visualizer': lrp_UNet3D_overfit_visualizer(config_data)
	elif config_data['visual_mode'] == 'lrp_UNet3D_filter_sweep_visualizer': lrp_UNet3D_filter_sweep_visualizer(config_data) 
	elif config_data['visual_mode'] == 'lrp_UNet3D_filter_sweep_0002_visualizer': lrp_UNet3D_filter_sweep_0002_visualizer(config_data) 
	elif config_data['visual_mode'] == 'lrp_UNet3D_filter_sweep_0003_visualizer': lrp_UNet3D_filter_sweep_0003_visualizer(config_data) 
	else: raise Exception('Invalid mode!')
	return

def lrp_UNet3D_filter_sweep_0003_visualizer(config_data, verbose=0):
	print("lrp_UNet3D_filter_sweep_0003_visualizer()")
	model_label_names=[
		'UNet3D_AXXXS1',
		'UNet3D_AXXXS2',
	]

	model_label_names_collection = {
		'[48,48,19]': model_label_names,
		'[96,96,19]': ['UNet3D_BXXXS1','UNet3D_BXXXS2'],
		'[192,192,19]': ['UNet3D_XXXXS5','UNet3D_XXXXS6'] # ,'UNet3D_XXXXX3' failure
	}

	color_list = {
		'[48,48,19]': 'b',
		'[96,96,19]': 'orange',
		'[192,192,19]': 'g'
	}
	
	print("="*80)
	print('fraction_pass_filter')
	vh.lrp_UNet3D_filter_sweep_0003_visualizer_aux(config_data,model_label_names_collection, 'fraction_pass_filter', color_list, verbose=0)

	print("="*80)
	print('fraction_clamp_filter')
	vh.lrp_UNet3D_filter_sweep_0003_visualizer_aux(config_data,model_label_names_collection, 'fraction_clamp_filter', color_list, verbose=00)
	# plt.show()

def lrp_UNet3D_filter_sweep_0002_visualizer(config_data, verbose=0):
	print("lrp_UNet3D_filter_sweep_0002_visualizer()")
	# config_data['model_label_name']
	model_label_names=[
		'UNet3D_AXXXS1',
		'UNet3D_AXXXS2',
		'UNet3D_AXXXX3',
		'UNet3D_AXXXX4'
	]

	model_label_names_collection = {
		'[48,48,19]': model_label_names,
		'[96,96,19]': ['UNet3D_BXXXS1','UNet3D_BXXXS2','UNet3D_BXXXX3','UNet3D_BXXXX4'],
		'[144,144,19]': ['UNet3D_CXXXX1','UNet3D_CXXXX2','UNet3D_CXXXX3','UNet3D_CXXXX4'],
		'[192,192,19]': ['UNet3D_XXXXX1','UNet3D_XXXXX2','UNet3D_XXXXX4',\
			'UNet3D_XXXXS5','UNet3D_XXXXS6'] # ,'UNet3D_XXXXX3' failure
	}
	
	color_list = {
		'[48,48,19]': 'w',
		'[96,96,19]': 'b',
		'[144,144,19]': 'g',
		'[192,192,19]': 'r'
	}

	vh.lrp_UNet3D_filter_sweep_0002_visualizer_aux(config_data,model_label_names_collection, 'fraction_pass_filter', color_list, verbose=0)
	print("\n")
	print("="*100)
	print("\n")
	vh.lrp_UNet3D_filter_sweep_0002_visualizer_aux(config_data,model_label_names_collection, 'fraction_clamp_filter', color_list, verbose=0)
	plt.show()


def lrp_UNet3D_filter_sweep_visualizer(config_data):
	print('lrp_UNet3D_filter_sweep_visualizer()')
	filter_sub_mode = config_data['LRP']['filter_sweeper']['submode']

	case_type = 'training'
	case_number = config_data['misc']['case_number']
	data_modalities = config_data['data_modalities']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	lrp_dir = os.path.join(model_dir,'lrp_filter_sweep_mode_' + filter_sub_mode )
	print("  lrp_dir:%s"%(str(lrp_dir)))

	from pipeline.lrp import get_modalities_0001
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)

	dict_name = 'lrp_dict_' + case_type + '_' + str(case_number) + '.lrpd'
	lrp_fullpath = os.path.join(lrp_dir,dict_name)
	output_dictionary = vh.load_lrp_sweep(lrp_fullpath)

	from dataio.dataISLES2017 import ISLES2017mass
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
	x1, _ = ISLESDATA.load_type0001_get_raw_input(one_case, modalities_dict)	

	from utils.vis0001 import SlidingVisualizerMultiPanelLRP
	vis = SlidingVisualizerMultiPanelLRP()
	vis.vis_filter_sweeper(config_data, output_dictionary, x1, case_number)

def lrp_UNet3D_overfit_visualizer(config_data):
	print("lrp_UNet3D_overfit_visualizer()")
	normalization_mode = str(config_data['LRP']['relprop_config']['normalization'])
	case_type = 'training'
	case_number = config_data['misc']['case_number']
	data_modalities = config_data['data_modalities']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	lrp_dir = os.path.join(model_dir,'output_lrp_' + normalization_mode)
	lrp_fullpath = os.path.join(lrp_dir,'yR_'+case_type+'_'+str(case_number)+'.lrp')

	print("  model_dir:%s"%(str(model_dir)))
	print("  data_modalities:%s"%(str(data_modalities)))
	print("  case_number:%s"%(str(case_number)))
	print("  normalization_mode:%s"%(str(normalization_mode)))
	pkl_file = open(lrp_fullpath, 'rb')
	y, Rc = pickle.load(pkl_file)
	pkl_file.close() 
	print("  Rc.shape = %s [%s]\n  y.shape = %s [%s]"%(str(Rc.shape),str(type(Rc)),str(y.shape),str(type(y))))
	
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	one_case = ISLESDATA.load_one_case(case_type, str(case_number), data_modalities)

	vis = vi.SlidingVisualizer()
	vis.do_show = True
	# vis.vis2_lrp_training(one_case,y,Rc,data_modalities)
	args = (one_case,y,Rc,data_modalities)
	p = Pool(2)
	p.map(vis.vis_lrp_training, [args+(1,),args+(2,)] )

	

def plot_loss_0001(config_data):
	print('visual.py. plot_loss_0001()')

	model_label_name_list = ['UNet3D_YXXXXX','UNet3D_YXXXX2','UNet3Db_YXXXXX','UNet3Db_YXXXX2']
	epoch_list = [10,20,30,40]
	filename_list = ['crossentropyloss_tracker_'+str(i)+'.evalobj' for i in epoch_list]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	color_scheme = np.linspace(0,1,len(model_label_name_list))
	
	for i, model_label_name in enumerate(model_label_name_list):
		folder_path = os.path.join(config_data['working_dir'],config_data['relative_checkpoint_dir'])
		print("  model_label_name:%s"%(model_label_name))
		global_losses = []
		for filename in filename_list:
			full_file_path = os.path.join(folder_path,model_label_name,filename)
			cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=1).load_state(config_data, filename=full_file_path)
			for loss in cetracker.global_error: global_losses.append(loss)
			# print("    cetracker.global_error.shape :%s"%(str(np.array(cetracker.global_error).shape)))
		ax.plot(range(len(global_losses)),global_losses, color=(1-color_scheme[i],0,color_scheme[i]),label=model_label_name)
	ax.set_xlabel("n batches")
	ax.set_ylabel("loss")
	ax.legend()
	plt.title("Cross entropy losses")
	plt.tight_layout()
	plt.show()
