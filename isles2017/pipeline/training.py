from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
import utils.loss as uloss
import utils.evalobj as ev
import models.networks as net

import utils.optimizer as op
# DEBUG_TRAINING_CASE_NUMBERS = range(1,10) # default is None
# DEBUG_TRAINING_LOOP = 0
# # DEBUG_TRAINING_LABELS_RESHAPE = 0 # if memory usage too large without reshaping during tests. Set to "resize": "[124,124,9]" when testing
# DEBUG_TRAINING_DATA_LOADING = 0

def training_UNet3D(config_data):
	print("training_UNet3D()")
	case_numbers = range(1,49)
	if DEBUG_TRAINING_CASE_NUMBERS is not None: case_numbers = DEBUG_TRAINING_CASE_NUMBERS
	case_type = 'training'
	
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	ISLESDATA.load_many_cases(case_type, case_numbers, config_data)
	
	if DEBUG_TRAINING_DATA_LOADING: return
	
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=config_data['basic']['batch_size'], 
		shuffle=True)
	print("  trainloader loaded")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.UNet3D(no_of_input_channel=ISLESDATA.no_of_input_channels, with_LRP=True)

	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	this_net.training_cycle = this_net.training_cycle + 1
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	criterion = nn.CrossEntropyLoss()
	optimizer = op.get_optimizer(this_net, config_data)
	cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=2)

	for i_epoch in range(config_data['basic']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
			labels = data[1].to(this_device).to(torch.float)

			x = x.permute(0,1,4,3,2)
			labels = labels.permute(0,3,2,1)
			if DEBUG_TRAINING_LOOP: outputs = this_net.forward_debug(x);break

			outputs = this_net(x).contiguous() # .view((config_data['basic']['batch_size'],number_of_classes,)+resize_shape)
				
			loss = criterion(outputs, labels.to(torch.int64))
			cetracker.store_loss(loss) # tracking progress
			loss.backward()
			optimizer.step()
		this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data,no_of_data_processed=str(i+1) )

		this_net.save_models(this_net, config_data)
		this_net.clear_up_models(this_net,config_data, keep_at_most_n_latest_models=config_data['basic']['keep_at_most_n_latest_models'])
	cetracker.save_loss_plot(label_tag = str(this_net.latest_epoch))
