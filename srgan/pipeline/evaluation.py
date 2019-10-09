from utils.utils import *
from pipeline.evaluation_aux import *
from dataio.cifar2loader import Cifar10Data
import models.networks as net
import utils.evalobj as ev

def evaluate_small_cnn(config_data):
	print("pipeline/evaluation.py. evaluate_small_cnn()")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.SmallCNN()
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model: %s"%(str(main_model_fullpath)))
	else: raise Exception('Model to load %s not found'%(main_model_fullpath))

	# (1) Training set (overfitting).
	#
	#
	CFtraining = Cifar10Data()
	if DEBUG_EVALUATION_SIZE_PER_RAW_TRAINING_BATCH is not None: CFtraining.size_per_raw_batch = DEBUG_EVALUATION_SIZE_PER_RAW_TRAINING_BATCH
	CFtraining.load_data(config_data)
	trainloader = DataLoader(dataset=CFtraining, num_workers=0, batch_size=1, shuffle=False)
	print("=== trainloader loaded ===")
	debug_signal = evaluate_small_cnn_main_loop(config_data, model_dir, trainloader, this_net, 'evaluation_on_training_set', eval_mode=True); 
	if debug_signal == 1: return


	# (2) Testing .
	#
	#
	CFtesting = Cifar10Data()
	if DEBUG_EVALUATION_SIZE_PER_RAW_TESTING_BATCH is not None: CFtesting.size_per_raw_batch = DEBUG_EVALUATION_SIZE_PER_RAW_TESTING_BATCH
	CFtesting.load_data(config_data, split = 'test')
	testloader = DataLoader(dataset=CFtesting, num_workers=0, batch_size=1, shuffle=False)
	print("=== testloader loaded ===")
	debug_signal = evaluate_small_cnn_main_loop(config_data, model_dir, testloader, this_net, 'evaluation_on_testing_set', eval_mode=True); 
	if debug_signal == 1: return

def evaluate_small_cnn_main_loop(config_data, model_dir, loader, this_net, report_name, eval_mode=False):
	from models.SmallCNN import no_of_classes

	if eval_mode: this_net.eval()
	pa = ev.PredictionAcc()
	pa.setup_classification_matrix(no_of_classes)
	for i, data in enumerate(loader, 0):
		x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
		labels = data[1].to(this_device).to(torch.float)	
		y_ot = int(labels.item())

		outputs = this_net(x).contiguous() 
		y = torch.argmax(outputs).item()
		if DEBUG_EVALUATION_LOOP: evaluate_small_cnn_debug0001(outputs, y, y_ot); return 1

		pa.update_prediction_acc_0001(y, y_ot)
	
	if not DEBUG_EVALUATION_LOOP:
		pa.total_number_of_data_evaluated = i + 1
		pa.process_evaluation_0001(model_dir, this_net, report_name=report_name)

	return 0

def evaluate_small_cnn_gs(config_data,config_data_gs):
	print("pipeline/evaluation.py. evaluate_small_cnn_gs()")

	def find_model(config_data):
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
		this_net = net.SmallCNN()
		if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model: %s"%(str(main_model_fullpath)))
		else: raise Exception('Model to load %s not found'%(main_model_fullpath))
		return this_net, model_dir

	this_net,_ = find_model(config_data)
	gs_net,model_dir = find_model(config_data_gs)

	# (1) Training set (overfitting).
	#
	#
	CFtraining = Cifar10Data()
	if DEBUG_EVALUATION_SIZE_PER_RAW_TRAINING_BATCH is not None: CFtraining.size_per_raw_batch = DEBUG_EVALUATION_SIZE_PER_RAW_TRAINING_BATCH
	CFtraining.load_data(config_data)
	trainloader = DataLoader(dataset=CFtraining, num_workers=0, batch_size=1, shuffle=False)
	print("=== trainloader loaded ===")
	debug_signal = evaluate_small_cnn_main_gs_loop(model_dir, trainloader, this_net, gs_net, 'evaluation_on_training_set', eval_mode=True); 
	if debug_signal == 1: return


	# (2) Testing .
	#
	#
	CFtesting = Cifar10Data()
	if DEBUG_EVALUATION_SIZE_PER_RAW_TESTING_BATCH is not None: CFtesting.size_per_raw_batch = DEBUG_EVALUATION_SIZE_PER_RAW_TESTING_BATCH
	CFtesting.load_data(config_data, split = 'test')
	testloader = DataLoader(dataset=CFtesting, num_workers=0, batch_size=1, shuffle=False)
	print("=== testloader loaded ===")
	debug_signal = evaluate_small_cnn_main_gs_loop(model_dir, testloader, this_net, gs_net, 'evaluation_on_testing_set', eval_mode=True); 
	if debug_signal == 1: return


def evaluate_small_cnn_main_gs_loop(model_dir, loader, this_net, gs_net, report_name, eval_mode=False):
	from models.SmallCNN import no_of_classes

	if eval_mode: this_net.eval()
	pa = ev.PredictionAcc()
	pa.setup_classification_matrix(no_of_classes)
	for i, data in enumerate(loader, 0):
		x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
		labels = data[1].to(this_device).to(torch.float)	
		y_ot = int(labels.item())

		outputs = this_net(x).contiguous() - gs_net(x).contiguous()
		# if i%20==0: print("      %s | %s"%(str(this_net(x).contiguous().detach().cpu().numpy()),str(gs_net(x).contiguous().detach().cpu().numpy())))
		y = torch.argmax(outputs).item()
		if DEBUG_EVALUATION_LOOP: evaluate_small_cnn_debug0001(outputs, y, y_ot); return 1

		pa.update_prediction_acc_0001(y, y_ot)
	
	if not DEBUG_EVALUATION_LOOP:
		pa.total_number_of_data_evaluated = i + 1
		pa.process_evaluation_0001(model_dir, this_net, report_name=report_name)

	return 0