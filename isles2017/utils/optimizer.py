from utils.utils import *


# optimizer = optim.Adam(this_net.parameters(), lr=0.0002, betas=(0.5, 0.9))
def get_optimizer(this_net, config_data, verbose=10):
	if verbose>=10: print("  utils/optimizer.py. get_optimizer().")
	if config_data['learning']['mechanism'] == 'SGD':
		if verbose>=10: print("    Setting up optimizer: SGD")
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], 
			momentum=config_data['learning']['momentum'], 
			weight_decay=config_data['learning']['weight_decay'])	
	elif config_data['learning']['mechanism'] == 'adam':
		if verbose>=10: print("    Setting up optimizer: adam")
		optimizer = optim.Adam(this_net.parameters(), lr=config_data['learning']['learning_rate'],
			 betas=config_data['learning']['betas'])
	return optimizer