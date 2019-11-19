from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
import utils.loss as uloss
import utils.evalobj as ev
import utils.optimizer as op
import models.networks as net
import models.networks_utils as nut

from pipeline.training_aux import *  

case_type = 'training'
case_numbers = range(1,49)
if DEBUG_TRAINING_CASE_NUMBERS is not None: case_numbers = DEBUG_TRAINING_CASE_NUMBERS

def training_UNet3D(config_data):
	print("training_UNet3D()")
	
	ISLESDATA, trainloader, TERMINATE_SIGNAL = training_UNet3D_load_isles2017(case_type, case_numbers, config_data)
	if TERMINATE_SIGNAL: return

	this_net = nut.get_UNet3D_version(config_data, ISLESDATA.no_of_input_channels, training_mode=config_data['training_mode'] , training=True)
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	criterion, optimizer, cetracker = training_UNet3D_setup_tools(this_net, config_data)

	for i_epoch in range(config_data['basic']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			x, labels = training_UNet3D_micro_data_prep(data)
			if DEBUG_TRAINING_LOOP: outputs = this_net.forward_debug(x);break

			outputs = this_net(x, save_for_relprop=False).contiguous()

			loss, DEBUG_SIGNAL_LOSS = uloss.training_UNet3D_compute_loss(criterion, outputs, labels.to(torch.int64),config_data, this_net, x)
			if DEBUG_SIGNAL_LOSS: return
			cetracker.store_loss(loss.clone()) # tracking progress
			loss.backward()
			optimizer.step()
		this_net.stop_timer()
		
		if DEBUG_TRAINING_LOOP: return
		this_net = this_net.post_process_sequence(this_net, config_data, no_of_data_processed=str(i+1))
	cetracker.save_loss_plot(label_tag=str(this_net.latest_epoch))
	cetracker.save_state(config_data, filename='crossentropyloss_tracker_' + str(this_net.latest_epoch) + '.evalobj')



