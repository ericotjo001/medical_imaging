from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
from pipeline.evaluation import get_modalities_0001
from pipeline.evaluation_header import generic_data_loading
from utils.vis2 import SlidingVisualizer2
import utils.loss as uloss
import models.networks_utils as nut

INFO_VIS_DIFF2 = """Use this mode by manually changing parameters liberally.

for config_data['visual_mode'] == 'diffgen_0002'

Format of this visual modes:
  python main.py --mode visual --subsubmode 0001
    > print dice scores. Can show by range.
  python main.py --mode visual --subsubmode 0002 --case_number 1
    > Slider figure showing different modalities of the slices. 
  python main.py --mode visual --subsubmode 0003 
    > Plot dice by size. 
"""

pm = PrintingManager()

def visual_diff_gen_0002(config_data):
	console_case_number = config_data['console_case_number']
	console_subsubmode = config_data['console_subsubmode']
	print('visual_diff_gen_0002(). subsubmode:%s, console_case_number:%s'%(str(console_subsubmode),str(console_case_number)))

	if console_subsubmode is None:
		print(INFO_VIS_DIFF2)
	elif str(console_subsubmode) == '0001':
		visual_diff_gen_0002_0001(config_data)
	elif str(console_subsubmode) == '0002':
		visual_diff_gen_0002_0002(config_data)
	elif str(console_subsubmode) == '0003':
		from pipeline.visual_dice_by_size import visual_diff_gen_0002_0003
		visual_diff_gen_0002_0003(config_data)
	else:
		print('(!) Invalid mode')
		print(INFO_VIS_DIFF2)


def visual_diff_gen_0002_0001(config_data, visual_config=None):
	print('visual_diff_gen_0002_0001(). Show dice_scores.ev')
	
	###########################################################
	# adhoc. 
	# EXAMPLE 1
	visual_config = {
		'model_label_name':'UNet3D_SDGX21',
		'dice_range': (1., 0.),
		'dice_score_name':'dice_scores.ev'
	}
	config_data['dataloader']['resize'] = [48, 48, 19]

	# EXAMPLE 2
	# visual_config = {
	# 	'model_label_name':'UNet3D_XDGX22',
	# 	'dice_range': (0.1, 0.),
	# 	'dice_score_name':'dice_scores.ev'
	# }
	# config_data['dataloader']['resize'] = [192,192,19]
	###########################################################

	if visual_config is None:
		FILENAME = 'dice_scores.ev'
		UPPER_DICE_RANGE, LOWER_DICE_RANGE = 1., 0.
		MODEL_LABEL_NAME = config_data['model_label_name']
	else:
		FILENAME = visual_config['dice_score_name']
		UPPER_DICE_RANGE, LOWER_DICE_RANGE = visual_config['dice_range']
		MODEL_LABEL_NAME = visual_config['model_label_name']
	

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],MODEL_LABEL_NAME)
	fullpath = os.path.join(model_dir, FILENAME)

	pkl_file = open(fullpath, 'rb')
	ev = pickle.load(pkl_file)
	pkl_file.close()

	ev1 = ev['dice_scores0'] #eval_mode
	ev2 = ev['dice_scores1'] #eval_omde2, main evaluation mode
	ev3 = ev['dice_scores2'] # train mode
	
	print("evaluation in train mode:")
	for epoch, y in ev3.items():
		list_of_dice = [y1 for case_number, y1 in y.items()]
		print('  %4s:%s'%(str(epoch),str(np.mean(list_of_dice))))
		"""
		   1:0.2530941852303438
		   2:0.2704565012177756
		   3:0.3018786629965139
		   4:0.3063010393187057
			latest:0.3063010393187057
		"""

	for epoch_dice_1,epoch_dice_2,epoch_dice_3 in zip(ev1.items(),ev2.items(), ev3.items()):
		epoch, dice_dict1 = epoch_dice_1
		_, dice_dict2 = epoch_dice_2
		_, dice_dict3 = epoch_dice_3

		print('[%s] Dices by case_numbers in this range [%s,%s]'%(str(epoch),str(LOWER_DICE_RANGE),str(UPPER_DICE_RANGE)))
		for case_number in dice_dict1:
			dice1 = round(dice_dict1[case_number],7)
			dice2 = round(dice_dict2[case_number],7) # main evaluation mode
			dice3 = round(dice_dict3[case_number],7)
			if dice2 >= LOWER_DICE_RANGE and dice2 <= UPPER_DICE_RANGE:
				print('  %4s | %10s | %10s | %10s | '%(str(case_number),str(dice1),str(dice2),str(dice3)))

def visual_diff_gen_0002_0002(config_data, visual_config=None):
	tab_level=1 

	pm.printvm('visual_diff_gen_0002_0002(). Show images case by case.', tab_level=tab_level)
	this_case_number = config_data['console_case_number']

	################################################################
	# adhoc
	# EXAMPLE 1
	visual_config = {'model_label_name':'UNet3D_SDGX21'}
	config_data['dataloader']['resize'] = [48,48,19]

	# EXAMPLE 2
	# visual_config = {'model_label_name':'UNet3D_XDGX22'}
	# config_data['dataloader']['resize'] = [192,192,19]
	################################################################

	if visual_config is None:
		MODEL_LABEL_NAME = config_data['model_label_name']
	else:
		MODEL_LABEL_NAME = visual_config['model_label_name']
		config_data['model_label_name'] = MODEL_LABEL_NAME

	sv = SlidingVisualizer2(do_show=True)
	sv.canonical_modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP',6:'OT',7:'pred'}

	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
	model_type = "UNet3D_diff"
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],MODEL_LABEL_NAME)
	this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
	this_net.eval()

	one_case = prepare_simple_one_case_for_viewing(config_data, this_net, this_case_number, tab_level=tab_level+1)

	sv.vis1(one_case,)

def prepare_simple_one_case_for_viewing(config_data, this_net, this_case_number, tab_level=0):
	dice_loss = uloss.SoftDiceLoss()
	for_evaluation = generic_data_loading(config_data, case_numbers_manual=[int(this_case_number)], 
		case_type='training', verbose=0)
	for case_number in for_evaluation:
		# THERE SHOULD ONLY BE ONE case_number IN THIS MODE
		x, labels = for_evaluation[case_number]
		outputs = torch.argmax(this_net(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs_label = interp3d(outputs.clone().detach(),tuple(labels.shape), mode='nearest')
		d = dice_loss(outputs_label,labels , factor=1)
		dice_score = 1 - d.item()
		
		x1 = batch_interp3d(x,tuple(labels.shape[::-1]),mode='linear')
		x1 = x1.squeeze().detach().cpu().numpy()
		x1 = np.transpose(x1,(0,3,2,1))
		labels = labels.detach().cpu().numpy()
		outputs = outputs.detach().cpu().numpy()
		outputs_label = outputs_label.detach().cpu().numpy()
		pm.printvm('x1.shape:           %s'%(str(x1.shape)),tab_level=tab_level)
		pm.printvm('outputs.shape:      %s'%(str(outputs.shape)),tab_level=tab_level)
		pm.printvm('outputs_label.shape:%s'%(str(outputs_label.shape)),tab_level=tab_level)
		pm.printvm('dice_score:         %s'%(str(dice_score)),tab_level=tab_level)
	    # x.shape:            (6, 19, 48, 48)
	    # outputs.shape:      (48, 48, 19)
	    # outputs_label.shape:(192, 192, 19)

	one_case = {'imgobj': {}}
	modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP',6:'OT'}
	for mod_index, modality in modalities_dict.items():
		if modality == 'OT':
			one_case['imgobj'][modality] = labels
		else:
			one_case['imgobj'][modality] = x1[mod_index]
	one_case['imgobj']['pred'] = outputs_label
	return one_case