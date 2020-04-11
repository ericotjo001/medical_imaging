from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
from pipeline.evaluation import get_modalities_0001
from pipeline.evaluation_header import generic_data_loading
import utils.loss as uloss
import models.networks_utils as nut

pm = PrintingManager()

def cache_save(filename, data):
	output = open(filename, 'wb')
	pickle.dump(data, output)
	output.close()

def cache_load(filename):
	pkl_file = open(filename, 'rb')
	data1 = pickle.load(pkl_file)
	pkl_file.close()
	return data1

def visual_diff_gen_0002_0003(config_data):	
	filename = 'vdg23.cache'
	if not os.path.exists(filename):
		dict_by_model_name = {}

		visual_config = {'model_label_name':'UNet3D_SDGX21'}
		config_data['dataloader']['resize'] = [48,48,19]
		visual_diff_gen_0002_0003_one_model(config_data, dict_by_model_name, visual_config=visual_config)

		visual_config = {'model_label_name':'UNet3D_XDGX22'}
		config_data['dataloader']['resize'] = [192,192,19]
		visual_diff_gen_0002_0003_one_model(config_data, dict_by_model_name, visual_config=visual_config)

		pm.print_recursive_dict(dict_by_model_name,
			tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None)
		cache_save(filename, dict_by_model_name)

	else:
		print('loading cache! Delete cache if you want to rerun the data collection process.')
		dict_by_model_name = cache_load(filename)

	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	for model_name, frac_dice_data in dict_by_model_name.items():
		x = frac_dice_data['x_frac']
		x = np.array(x)
		xp = -np.log10(x)
		y = frac_dice_data['y_dice']
		y = np.array(y)

		X = [[x1] for x1 in xp]
		reg = LinearRegression().fit(X, y)
		r = reg.score(X, y)**0.5
		m, c = reg.coef_[0], reg.intercept_
		x0 = np.linspace(np.min(xp), np.max(xp), 250)
		y_pred = m*x0+c

		ax2.scatter(-np.log10(x),y, label='%s (%s)'%(str(model_name),str(round(r,3))), s=3)
		ax2.plot(x0,y_pred)
		print('%s: m:%s c:%s'%(str(model_name),str(m),str(c)))
	ax2.set_xlabel('-log(x)')
	ax2.set_ylabel('y')
	plt.legend()
	plt.show()

def visual_diff_gen_0002_0003_one_model(config_data, dict_by_model_name, visual_config=None):	
	tab_level=1 
	pm.printvm('visual_diff_gen_0002_0003_one_model(). Plot dice by size.', tab_level=tab_level)

	################################################################
	# adhoc
	# EXAMPLE 1
	# visual_config = {'model_label_name':'UNet3D_SDGX21'}
	# config_data['dataloader']['resize'] = [48,48,19]

	# EXAMPLE 2
	# visual_config = {'model_label_name':'UNet3D_XDGX22'}
	# config_data['dataloader']['resize'] = [192,192,19]
	################################################################

	if visual_config is None:
		MODEL_LABEL_NAME = config_data['model_label_name']
	else:
		MODEL_LABEL_NAME = visual_config['model_label_name']
		config_data['model_label_name'] = MODEL_LABEL_NAME

	dict_by_model_name = collect_dice_vs_size_data_across_models(dict_by_model_name, MODEL_LABEL_NAME, config_data,tab_level=tab_level+1)
	# pm.print_recursive_dict(dict_by_model_name,
	# 	tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None)

def collect_dice_vs_size_data_across_models(dict_by_model_name, MODEL_LABEL_NAME, config_data, tab_level=0):
	x_frac, y_dice = [], []
	case_numbers = range(1,49) # range(1,49)
	for this_case_number in case_numbers: # 1-48 are case numbers, though some are missing
		dice, frac = prepare_cases_for_dice_by_size(config_data, MODEL_LABEL_NAME, this_case_number, tab_level=tab_level)
		if dice is None: continue
		x_frac.append(frac)
		y_dice.append(dice)
	dict_by_model_name[MODEL_LABEL_NAME] = {
		'x_frac':x_frac, 'y_dice': y_dice
	}
	return dict_by_model_name

def prepare_cases_for_dice_by_size(config_data, MODEL_LABEL_NAME, this_case_number, tab_level=0):
	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
	model_type = "UNet3D_diff"
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],MODEL_LABEL_NAME)
	dice_loss = uloss.SoftDiceLoss()

	# we reinit the models for each run to reset the memory
	this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
	this_net.eval()

	for_evaluation = generic_data_loading(config_data, case_numbers_manual=[int(this_case_number)],
		case_type='training', verbose=0)
	if len(for_evaluation) == 0:
		return None, None
	for case_number in for_evaluation:
		# SET 1 CASE ONLY
		x, labels = for_evaluation[case_number] # labels -> groundtruth
		outputs = torch.argmax(this_net(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		d = dice_loss(outputs_label,labels , factor=1)
		dice_score = 1 - d.item()
		s = labels.reshape(-1)
		n_pixels_with_lesion = sum(s.detach().cpu().numpy())
		s  = np.array(list(s.shape))
		total_pixel = s[0]
		frac = n_pixels_with_lesion /total_pixel
		
		x = x.squeeze().detach().cpu().numpy()
		x = np.transpose(x,(0,3,2,1))
		labels = labels.detach().cpu().numpy()
		outputs = outputs.detach().cpu().numpy()
		outputs_label = outputs_label.detach().cpu().numpy()
		# pm.printvm('x.shape:            %s'%(str(x.shape)),tab_level=tab_level)
		# pm.printvm('outputs.shape:      %s'%(str(outputs.shape)),tab_level=tab_level)
		# pm.printvm('outputs_label.shape:%s'%(str(outputs_label.shape)),tab_level=tab_level)
		# pm.printvm('dice_score          :%s'%(str(dice_score)),tab_level=tab_level)
		# pm.printvm('frac                :%s'%(str(frac)),tab_level=tab_level)
	    # x.shape:            (6, 19, 48, 48)
	    # outputs.shape:      (48, 48, 19)
	    # outputs_label.shape:(192, 192, 19)
	return dice_score, frac