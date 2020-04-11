from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
import utils.loss as uloss
import utils.evalobj as ev
import utils.optimizer as op
import models.networks as net
import models.networks_utils as nut
from pipeline.training_aux import training_UNet3D_setup_tools

if DEBUG_VERSION:
	DEBUG_TRAINING_DIFF_DATA_LOOP = 0
	DEBUG_TRAINING_DIFF_LOOP = 0
	DEBUG_TRAINING_CASE_NUMBERS_DG = range(1,8)
else:
	DEBUG_TRAINING_DIFF_DATA_LOOP = False
	DEBUG_TRAINING_DIFF_LOOP = False # (bool)
	DEBUG_TRAINING_CASE_NUMBERS_DG = None

case_type = 'training'
case_numbers = range(1,49)
if DEBUG_TRAINING_CASE_NUMBERS_DG is not None: case_numbers = DEBUG_TRAINING_CASE_NUMBERS_DG

def training_UNet3D_diff(config_data, DATA_PER_EPOCH=10, defect_fraction=1.):
	print('training_UNet3D_diff()')
	
	config_data['basic']['batch_size'] = 1
	N_EPOCH = config_data['basic']['n_epoch']
	BATCH_SIZE = config_data['basic']['batch_size']
	CHANNEL_SIZE= len(config_data['data_modalities'])-1 #-1 for ground truth listed in the modalities

	from dataio.data_diffgen import DG3D
	dg = DG3D(unit_size=(192,192), depth=19) # this size will be interpolated to 3D (19,192,192)

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	ISLESDATA.load_many_cases(case_type, case_numbers, config_data)
	N_ISLES = len(ISLESDATA.x)
	# x,y = ISLESDATA.__getitem__(0)
	# print(x.shape, y.shape) # (6, 48, 48, 19) (48, 48, 19) # NOT RESHAPED YET

	this_net = nut.get_UNet3D_version(config_data, CHANNEL_SIZE, training_mode=config_data['training_mode'] , training=True)
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	criterion, optimizer, cetracker = training_UNet3D_setup_tools(this_net, config_data)

	for i_epoch in range(N_EPOCH):
		this_net.start_timer()
		print('epoch number in this run:%s'%(str(i_epoch)))
		for i in range(DATA_PER_EPOCH):
			
			random_index = np.random.randint(0,N_ISLES)
			
			x_healthy, x_unhealthy, y_lesion, y_healthy = dg.generate_data_batches_in_torch(
				channel_size=6, batch_size=BATCH_SIZE,resize=config_data['dataloader']['resize'])
			isles_mri = torch.tensor([ISLESDATA.__getitem__(random_index)[0]]).permute(0,1,4,3,2)
			isles_lesion = torch.tensor([ISLESDATA.__getitem__(random_index)[1]]).permute(0,3,2,1).to(torch.float) 

			if DEBUG_see_data_diffgen(DEBUG_TRAINING_DIFF_DATA_LOOP,x_healthy, x_unhealthy, y_lesion, y_healthy): return
			if DEBUG_TRAINING_DIFF_LOOP: 
				outputs = this_net.forward_debug(x_unhealthy.to(device=this_device).to(torch.float)); return

			x = isles_mri.to(device=this_device) + defect_fraction*x_unhealthy.to(device=this_device)
			x = x.to(torch.float)
			labels = ((isles_lesion.to(device=this_device) + y_lesion.to(device=this_device))>0.).to(torch.float) 
			optimizer.zero_grad()
			outputs = this_net(x, save_for_relprop=False).contiguous()
			loss, DEBUG_SIGNAL_LOSS = uloss.training_UNet3D_compute_loss(criterion, outputs, 
				labels.to(torch.int64),config_data, this_net, x)
			if DEBUG_SIGNAL_LOSS: return
			cetracker.store_loss(loss.clone()) # tracking progress
			loss.backward()
			optimizer.step()

		this_net.stop_timer()
		
		if DEBUG_TRAINING_DIFF_LOOP: return
		this_net = this_net.post_process_sequence(this_net, config_data, no_of_data_processed=str(i+1))
	cetracker.save_loss_plot(label_tag=str(this_net.latest_epoch))
	cetracker.save_state(config_data, filename='crossentropyloss_tracker_' + str(this_net.latest_epoch) + '.evalobj')

def DEBUG_see_data_diffgen(DEBUG_TRAINING_DIFF_DATA_LOOP,x_healthy, x_unhealthy, y_lesion, y_healthy):
	DEBUG_SIGNAL=False

	if DEBUG_TRAINING_DIFF_DATA_LOOP:
		DEBUG_SIGNAL = True
		print('DEBUG_see_data_diffgen()')
		print('  x_healthy.shape:%s'%(str(x_healthy.shape)))
		print('  x_unhealthy.shape:%s'%(str(x_unhealthy.shape)))
		print('  y_lesion.shape:%s sum=%s'%(str(y_lesion.shape),str(torch.sum(y_lesion))))
		print('  y_healthy.shape:%s sum=%s'%(str(y_healthy.shape),str(torch.sum(y_healthy))))

	return DEBUG_SIGNAL