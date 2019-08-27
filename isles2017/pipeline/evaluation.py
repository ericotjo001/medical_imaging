from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
from utils.evalobj import EvalObj
import utils.loss as uloss
import models.networks as net
import utils.metric as me

DEBUG_CASE_NUMBERS = None # range(1,10) # default is None
DEBUG_EVAL_LOOP = 0

n_NSC, n_NSCy = 2, 1 # number of nonspatial channel

"""
# Under construction 
def evaluation_segnet_overfit(config_data):
	print("evaluation_segnet_overfit()")
	
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.segnet()
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	generic_evaluation_overfit_0001(config_data, this_net,model_dir,main_model_fullpath)
"""

"""
# Under construction 
def evaluation_PSPNet_overfit(config_data):
	print("evaluation_PSPNet_overfit()")
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.PSPNet()

	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	
	ev = EvalObj()
	for_evaluation = generic_data_loading(config_data)

	dice_loss = uloss.SoftDiceLoss()
	for save_epoch in this_net.saved_epochs:
		artifact_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(save_epoch) + '.model')
		this_net.load_state(config_data)
		this_net.training = False
		this_net.eval()

		dice_list = []
		for case_number in for_evaluation:
			x, labels = for_evaluation[case_number] 
			outputs = torch.argmax(this_net(x).contiguous(),dim=1)
			ous = tuple(outputs.shape[1:])
			outputs = outputs.view(ous[::-1]).to(torch.float)
			outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
			
			d = dice_loss(outputs_label,labels , factor=1)
			dice_score = 1 - d.item()
			dice_list.append(dice_score)

		if DEBUG_EVAL_LOOP: break
		ev.get_dice_score_at_epoch(save_epoch, dice_list)
	ev.save_evaluation(config_data)
"""

def evaluation_UNet3D_overfit(config_data):
	print("evaluation_UNet3D_overfit()")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	model_type = 'UNet3D'
	generic_evaluation_overfit_0001(config_data,model_type,model_dir)

"""
# Under construction 
def evaluation_FCN8like_overfit(config_data):
	print('evaluation_FCN8like_overfit()')

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.FCN8like()
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	generic_evaluation_overfit_0001(config_data, this_net,model_dir,main_model_fullpath)
"""

def generic_evaluation_overfit_0001(config_data, model_type,model_dir):
	ev = EvalObj()
	ev2 = EvalObj() # anotehr dice computation
	ev3 = EvalObj() # same as ev, but without eval()

	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	if model_type == 'UNet3D': this_net = net.UNet3D() 
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")

	for_evaluation = generic_data_loading(config_data)
	dice_loss = uloss.SoftDiceLoss()
	
	for save_epoch in this_net.saved_epochs:
		if model_type == 'UNet3D': artifact_net = net.UNet3D() 
		artifact_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(save_epoch) + '.model')
		artifact_net.load_state_dict(torch.load(artifact_model_fullpath))
		artifact_net.eval()
		print("Looking at model after epoch %s"%(str(save_epoch)))
		
		dice_list = []
		dice_list2 = [] # Adding another dice for double checking
		dice_list3 = []

		for case_number in for_evaluation:
			x, labels = for_evaluation[case_number] 
			
			# 1.
			outputs = torch.argmax(artifact_net(x).contiguous(),dim=1)
			outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
			outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
			if not check_labels(outputs_label): raise Exception("Output labels not 1.0 or 0.0")
			if DEBUG_EVAL_LOOP: DEBUG_generic_evaluation_overfit_0001(x, labels, outputs, outputs_label); break
			
			d = dice_loss(outputs_label,labels , factor=1)
			dice_score = 1 - d.item()
			dice_list.append(dice_score)

			# 2.
			# Adding another dice for double checking
			outputs_label_reshape = outputs_label.reshape((1,)+outputs_label.shape)
			label_reshape = labels.reshape((1,)+labels.shape)
			dice_score2 = me.DSC(outputs_label_reshape,label_reshape).item()
			dice_list2.append(dice_score2)

			# 3.
			artifact_net.train()
			outputs = torch.argmax(artifact_net(x).contiguous(),dim=1)
			outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
			outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
			d3 = dice_loss(outputs_label,labels , factor=1)
			dice_score3 = 1 - d3.item()
			dice_list3.append(dice_score3)

		ev.get_dice_score_at_epoch(save_epoch, dice_list)
		# Adding another dice for double checking
		ev2.get_dice_score_at_epoch(save_epoch, dice_list2)
		ev3.get_dice_score_at_epoch(save_epoch, dice_list3)

	if DEBUG_EVAL_LOOP: return

	print("Looking at the latest model.")
	dice_list = []
	dice_list2 = []
	dice_list3 = []
	for case_number in for_evaluation:
		x, labels = for_evaluation[case_number] 
		this_net.eval()

		# 1.
		outputs = torch.argmax(this_net(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		
		d = dice_loss(outputs_label,labels , factor=1)
		dice_score = 1 - d.item()
		dice_list.append(dice_score)

		# 2.
		# Adding another dice for double checking
		outputs_label_reshape = outputs_label.reshape((1,)+outputs_label.shape)
		label_reshape = labels.reshape((1,)+labels.shape)
		dice_score2 = me.DSC(outputs_label_reshape,label_reshape).item()
		dice_list2.append(dice_score2)

		# 3.
		this_net.train()
		outputs = torch.argmax(this_net(x).contiguous(),dim=1)
		outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
		outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
		d3 = dice_loss(outputs_label,labels , factor=1)
		dice_score3 = 1 - d3.item()
		dice_list3.append(dice_score3)		 


	ev.get_dice_score_latest(dice_list)
	ev.save_evaluation(config_data,report_name='report.txt')

	# Adding another dice for double checking
	ev2.get_dice_score_latest(dice_list2)
	ev2.save_evaluation(config_data,report_name='report2.txt')

	ev3.get_dice_score_latest(dice_list3)
	ev3.save_evaluation(config_data,report_name='report3.txt')



def generic_data_loading(config_data):
	print("Calling generic_data_loading()")
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP'}
	case_type = 'training'
	case_numbers = range(1,49)
	normalize = True
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	resize_shape = tuple(config_data['dataloader']['resize'])

	for_evaluation = {}
	for case_number in case_numbers:
		one_case = ISLESDATA.load_one_case(case_type, str(case_number), canonical_modalities_label)
		if one_case is None: continue
		s = one_case['imgobj']['ADC'].shape
		x1 = np.zeros(shape=(6,)+resize_shape)
		labels = one_case['imgobj']['OT']

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
		
		#x1 is now C,W,H,D
		x1s = x1.shape
		x = torch.tensor([x1]).to(torch.float).to(device=this_device)
		# print("x.shape:%s"%(str(x.shape)))
		x = x.permute(0,1,4,3,2)
		# print("x.shape after permute :%s"%(str(x.shape)))
		labels = torch.tensor(labels).to(torch.int64).to(device=this_device)
		for_evaluation[case_number] = [x,labels]
	print("data loaded for evaluation!")
	return for_evaluation

def DEBUG_generic_evaluation_overfit_0001(x, labels, outputs, outputs_label):
	print("DEBUG_generic_evaluation_overfit_0001()")
	print("  x.shape:%s"%(str(x.shape)))
	print("  labels.shape:%s"%(str(labels.shape)))
	print("  outputs.shape:%s"%(str(outputs.shape)))
	print("  outputs_label.shape:%s"%(str(outputs_label.shape)))

def check_labels(y):
	temp = y.detach().cpu().numpy().reshape(-1)
	unique_list = list(sorted(set(temp)))
	if DEBUG_EVAL_LOOP: print("  unique_list:",unique_list)
	for possible_list in [[0.],[1.],[0.,1.]]:
		if np.all(unique_list==possible_list): return True
	return False

#####################################
# the following is just for testing
#####################################

"""
# READY TO DEPRECATE

def evaluation_basic_1_overfit(config_data):
	print('evaluation_basic_1_overfit()')
	ev = EvalObj()

	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0001(case_type, case_numbers, config_data,normalize=True)
	testloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=1, shuffle=False)

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.CNN_basic(device=this_device)
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	else: raise Exception('Specified model does not exist! Typically it is in the checkpoint folder.')

	dice_loss = uloss.SoftDiceLoss()
	for save_epoch in this_net.saved_epochs:
		artifact_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(save_epoch) + '.model')
		this_net.load_state(config_data)
		
		dice_list = []
		for i, data in enumerate(testloader, 0):
			x = data[0].to(this_device).to(torch.float)
			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels = data[1].to(this_device).to(torch.float)
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
			
			outputs = torch.argmax(this_net(x),dim=1)

			if DEBUG_EVAL_LOOP: debug_evaluation_basic_1_overfit(x,labels,outputs) ;break

			d = dice_loss(outputs, labels, factor = 1)
			dice_score = 1 - d.item()
			dice_list.append(dice_score)
		if DEBUG_EVAL_LOOP: break
		ev.get_dice_score_at_epoch(save_epoch, dice_list)
	ev.save_evaluation(config_data)

def evaluation_FCN_1_overfit(config_data):
	print("evaulation_FCN_1_overfit()")

	ev = EvalObj()
	
	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0002(case_type, case_numbers, config_data,normalize=True)
	testloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=1, shuffle=False)
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.FCN(device=this_device)
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	else: raise Exception('Specified model does not exist! Typically it is in the checkpoint folder.')

	dice_loss = uloss.SoftDiceLoss()
	
	resize_shape = tuple(config_data['FCN_1']['resize'])
	for save_epoch in this_net.saved_epochs:
		artifact_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.' + str(save_epoch) + '.model')
		this_net.load_state(config_data)
		
		dice_list = []
		for i, data in enumerate(testloader, 0):
			x = data[0].to(this_device).to(torch.float)
			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels = data[1].to(this_device).to(torch.float)
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
			
			outputs = torch.argmax(this_net(x),dim=1).to(torch.float)
			if DEBUG_EVAL_LOOP: debug_evaluation_FCN_1_overfit(x,outputs, labels); break

			outputs_label = batch_no_channel_interp3d(outputs,tuple(labels.shape)[1:], mode='nearest').to(device=this_device)

			d = dice_loss(outputs_label, labels, factor = 1)
			dice_score = 1 - d.item()
			dice_list.append(dice_score)
		if DEBUG_EVAL_LOOP: break
		ev.get_dice_score_at_epoch(save_epoch, dice_list)
	ev.save_evaluation(config_data)

def debug_evaluation_basic_1_overfit(x,labels,outputs):
	print("===== debug_evaluation_basic_1_overfit =====")
	print("x.shape             = %s"%(str(x.shape)))
	print("labels.shape        = %s"%(str(labels.shape)))
	print("outputs.shape       = %s"%(str(outputs.shape)))
	print("============================================")

def debug_evaluation_FCN_1_overfit(x,outputs, labels):
	print("===== debug_evaluation_FCN_1_overfit =======")
	outputs_label = batch_no_channel_interp3d(outputs,tuple(labels.shape)[1:], mode='nearest')
	
	print("x.shape             = %s"%(str(x.shape)))
	print("outputs.shape       = %s"%(str(outputs.shape)))
	print("outputs_label.shape (after reshape) = %s"%(str(outputs_label.shape)))
	print("============================================")

def debug_evaluation_FCN8like_overfit(x, model, labels):
	print("===== debug_evaluation_FCN_1_overfit =======")
	print("x.shape             = %s"%(str(x.shape)))
	print("labels.shape        = %s"%(str(labels.shape)))
	outputs = torch.argmax(model(x).contiguous(),dim=1)
	ous = tuple(outputs.shape[1:])
	outputs = outputs.view(ous[::-1]).to(torch.float)
	print("outputs.shape       = %s"%(str(tuple(outputs.shape))))
	outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
	
	print("outputs_label.shape = %s"%(str(outputs_label.shape)))
	print("============================================")

"""