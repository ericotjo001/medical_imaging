from utils.utils import *

def DEBUG_training_small_cnn_0001(DEBUG_TRAINING_LOOP, this_net, x, labels):
	DEBUG_signal = 0
	if DEBUG_TRAINING_LOOP==1:
		DEBUG_signal = 1
		outputs = this_net.forward_debug(x);
		print("  outputs:%s outputs.shape:%s"%(str(outputs), str(outputs.shape)))
		print("  labels.to(torch.int64):%s labels.to(torch.int64).shape:%s"%(str(labels.to(torch.int64)), str(labels.to(torch.int64).shape)))
	return DEBUG_signal

# def DEBUG_training_small_cnn_0001gen(DEBUG_TRAINING_LOOP, this_net, gen_net, x, x_gen, labels, labels_gen):
# 	DEBUG_signal = 0
# 	if DEBUG_TRAINING_LOOP==1: 
# 		print("DEBUG_training_small_cnn_0001gen().")
# 		DEBUG_signal = DEBUG_training_small_cnn_0001(DEBUG_TRAINING_LOOP, this_net, x, labels)
# 	elif DEBUG_TRAINING_LOOP==2:
# 		print("DEBUG_training_small_cnn_0001gen().")
# 		DEBUG_signal = 1
# 		outputs = gen_net.forward_debug(x_gen);

# 		print("gen_net. forward2()")
# 		for j in [1,2,3,4]:
# 			print("  gen_net.forward2(x_gen,pos=%s.shape = %s"%(str(j),str(gen_net.forward2(x_gen,pos=j).shape)))

# 		print("  outputs.shape:%s"%(str(outputs.shape)))
# 		print("  labels_gen.to(torch.int64).shape:%s"%(str(labels_gen.to(torch.int64).shape)))
# 	return DEBUG_signal

# def adjust_config_data_for_helper_network(config_data):
# 	config_data_gen = config_data.copy()
# 	config_data_gen['model_label_name'] = config_data_gen['model_label_name'] + "_gen"
# 	print("  adjust_config_data_for_helper_network()")
# 	print("    config_data_gen['model_label_name']=%s"%(str(config_data_gen['model_label_name'])))
# 	print("    config_data['model_label_name']    =%s"%(str(config_data['model_label_name'])))
# 	config_data_gen['learning'] = config_data_gen['learning_for_discriminator']
# 	return config_data_gen

def adjust_config_data_for_helper_network(config_data, suffix="_2"):
	config_data2 = config_data.copy()
	config_data2['model_label_name'] = config_data2['model_label_name'] + suffix
	return config_data2

