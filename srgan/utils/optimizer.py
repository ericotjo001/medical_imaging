from utils.utils import *

# optimizer = optim.Adam(this_net.parameters(), lr=0.0002, betas=(0.5, 0.9))
def get_optimizer(this_net, config_data):
	if config_data['learning']['mechanism'] == 'SGD':
		print("  Setting up optimizer: SGD")
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], 
			momentum=config_data['learning']['momentum'], 
			weight_decay=config_data['learning']['weight_decay'])	
	elif config_data['learning']['mechanism'] == 'adam':
		print("  Setting up optimizer: adam")
		optimizer = optim.Adam(this_net.parameters(), lr=config_data['learning']['learning_rate'],
			 betas=config_data['learning']['betas'])
	return optimizer

# def get_optimizer_for_discriminator(this_net, config_data):
# 	if config_data['learning_for_discriminator']['mechanism'] == 'SGD':
# 		print("  Setting up optimizer: SGD")
# 		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning_for_discriminator']['learning_rate'], 
# 			momentum=config_data['learning_for_discriminator']['momentum'], 
# 			weight_decay=config_data['learning_for_discriminator']['weight_decay'])	
# 	elif config_data['learning_for_discriminator']['mechanism'] == 'adam':
# 		print("  Setting up optimizer: adam")
# 		optimizer = optim.Adam(this_net.parameters(), lr=config_data['learning_for_discriminator']['learning_rate'],
# 			 betas=config_data['learning_for_discriminator']['betas'])
# 	return optimizer