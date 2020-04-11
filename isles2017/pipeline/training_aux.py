from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
import utils.loss as uloss
import utils.evalobj as ev
import utils.optimizer as op
import models.networks as net
import models.networks_utils as nut

def training_UNet3D_load_isles2017(case_type, case_numbers, config_data):
	TERMINATE_SIGNAL = False
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	ISLESDATA.load_many_cases(case_type, case_numbers, config_data)

	if DEBUG_TRAINING_DATA_LOADING: 
		TERMINATE_SIGNAL = True
		trainloader = None
	else:
		trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, batch_size=config_data['basic']['batch_size'], shuffle=True)
		print("  trainloader loaded")
	return ISLESDATA, trainloader, TERMINATE_SIGNAL

def training_UNet3D_setup_tools(this_net, config_data):
	criterion = nn.CrossEntropyLoss()
	optimizer = op.get_optimizer(this_net, config_data)
	cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=2)
	return criterion, optimizer, cetracker

def training_UNet3D_micro_data_prep(data):
	x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
	labels = data[1].to(this_device).to(torch.float)

	x = x.permute(0,1,4,3,2)
	labels = labels.permute(0,3,2,1)
	return x, labels

def DEBUG_training_UNet3Db_filter_optim_loss(DEBUG_TRAINING_LOOP_LOSS, this_net, criterion, outputs, labels, mse_target_objective):
	DEBUG_SIGNAL = False
	if DEBUG_TRAINING_LOOP_LOSS:
		DEBUG_SIGNAL = True
		print("  DEBUG_TRAINING_LOOP_LOSS().")
		main_loss = criterion(outputs, labels.to(torch.int64)) 
		print("    main_loss:   %s"%(str(main_loss))) # is a tensor scalar
		t1 = this_net.lens.cvl.weight.data
		# s1 = (torch.sum((t1.reshape(-1))**2)**0.5)/t1.numel(); s1=s1.detach().cpu().numpy()
		# print("   t1.shape:%s root mean sqrsum = %s"%(str(t1.shape),str(s1)))
		filter_loss = uloss.filter_mse_loss(t1,mse_target_objective)
		print("    filter_loss: %s"%(str(filter_loss)))
		loss = main_loss + filter_loss
		print("    sum of loss: %s"%(str(loss)))

	return DEBUG_SIGNAL

