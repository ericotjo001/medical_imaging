from utils.utils import *
from pipeline.pipeline_aux import *
from dataio.cifar2loader import Cifar10Data
import utils.evalobj as ev
import models.networks as net
import utils.optimizer as op
from pipeline.training_aux import *


def training_small_cnn(config_data):
	print("pipeline/training.py. training_cnn().")

	data_dir = config_data['data_directory']['cifar10']
	CFDATA = Cifar10Data()
	if DEBUG_TRAINING_SIZE_PER_RAW_BATCH is not None: CFDATA.size_per_raw_batch = DEBUG_TRAINING_SIZE_PER_RAW_BATCH
	CFDATA.load_data(config_data)

	trainloader = DataLoader(dataset=CFDATA, num_workers=0, 
		batch_size=config_data['basic']['batch_size'], shuffle=True)
	print("  trainloader loaded")

	this_net = net.SmallCNN()
	this_net = this_net.perform_routine(this_net, config_data)
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

			if DEBUG_TRAINING_DATA_LOADER_PRINT: training_cnn_print_data(x, labels)
			if DEBUG_training_small_cnn_0001(DEBUG_TRAINING_LOOP, this_net, x, labels): break
			
			outputs = this_net(x).contiguous() 
			loss = criterion(outputs, labels.to(torch.int64))
			cetracker.store_loss(loss) # tracking progress
			loss.backward()
			optimizer.step()
		this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net = this_net.perform_routine_end(this_net, config_data, no_of_data_processed=str(i+1))
		
	cetracker.save_loss_plot(label_tag = str(this_net.latest_epoch))

def training_sr_small_cnn(config_data):
	import pipeline.training_gs as gs
	print("pipeline/training.py. training_sr_small_cnn()")
	print("sr: self-reflection. v2")

	data_dir = config_data['data_directory']['cifar10']
	CFDATA = Cifar10Data()
	if DEBUG_TRAINING_SIZE_PER_RAW_BATCH is not None: CFDATA.size_per_raw_batch = DEBUG_TRAINING_SIZE_PER_RAW_BATCH
	CFDATA.load_data(config_data, as_categorical=False)
	dict_class_by_index = CFDATA.create_dictionary_of_data_indices_sorted()

	trainloader = DataLoader(dataset=CFDATA, num_workers=0, batch_size=config_data['basic']['batch_size'], shuffle=True)
	print("  trainloader loaded")

	this_net = net.SmallCNN()
	this_net = this_net.perform_routine(this_net, config_data)
	net.count_parameters(this_net)
	criterion = nn.CrossEntropyLoss()
	optimizer = op.get_optimizer(this_net, config_data)
	cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=2)

	this_net2 = net.SmallCNN()
	config_data2 = adjust_config_data_for_helper_network(config_data)
	this_net2 = this_net2.perform_routine(this_net2, config_data2)
	net.count_parameters(this_net2)
	criterion2 = nn.CrossEntropyLoss()
	optimizer2 = op.get_optimizer(this_net2, config_data2)
	cetracker2 = ev.CrossEntropyLossTracker(config_data2, display_every_n_minibatchs=2)

	for i_epoch in range(config_data['basic']['n_epoch']):
		this_net.start_timer()
		this_net2.start_timer()
		count1, count2 = 0, 0
		for i, data in enumerate(trainloader, 0):
			x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
			labels = data[1].to(this_device).to(torch.float)

			if DEBUG_TRAINING_DATA_LOADER_PRINT: training_cnn_print_data(x, labels)
			# if DEBUG_training_small_cnn_0001gen(DEBUG_TRAINING_LOOP, this_net, gen_net, x, x_gen, labels, labels_gen): break
			
			if i%2==0:
				optimizer.zero_grad()	
				outputs = this_net(x).contiguous() 
				loss = criterion(outputs, labels.to(torch.int64))
				cetracker.store_loss(loss) # tracking progress
				loss.backward()
				optimizer.step()
				count1 +=1
			else:
				optimizer2.zero_grad()	
				outputs = this_net2(x).contiguous() 
				loss2 = criterion2(outputs, labels.to(torch.int64))
				cetracker2.store_loss(loss2) # tracking progress
				loss2.backward()
				optimizer2.step()
				count2+=1

		this_net.stop_timer()
		this_net2.stop_timer()

		if DEBUG_TRAINING_LOOP: break
		this_net = this_net.perform_routine_end(this_net, config_data, no_of_data_processed=str(count1))
		this_net2 = this_net2.perform_routine_end(this_net2, config_data2, no_of_data_processed=str(count2))
		
	cetracker.save_loss_plot(label_tag = str(this_net.latest_epoch))
	cetracker2.save_loss_plot(label_tag = str(this_net2.latest_epoch))


	gs_net = net.SmallCNN()
	config_data_gs = adjust_config_data_for_helper_network(config_data, suffix='_gs')
	gs_net = gs_net.perform_routine(gs_net, config_data_gs)
	gs_net.load_state(config_data2) # load from this_net2 
	gs_net = gs.gram_schmidt_process(gs_net, this_net, config_data_gs) # cut away the part of this_net