from utils.utils import *


def create_discriminator_config(config_data):
	config_data_discriminator = config_data.copy()
	config_data_discriminator['model_label_name'] = config_data['model_label_name'] + '_discriminator'
	# print(config_data['model_label_name'])
	# print(config_data_discriminator['model_label_name'])
	return config_data_discriminator


def training_sr_small_cnn_prepare_fake_prediction(x, labels):
	x_fake = torch.tensor(np.random.uniform(0,1,size=x.shape))
	y_fake = np.zeros(shape=labels.shape)
	for j in range(len(y_fake)): y_fake[j][np.random.randint(len(y_fake[j]))]=1
	y_fake = torch.tensor(y_fake)
	return x_fake ,y_fake

def training_sr_small_cnn_prepare_fake_groundtruth(this_discriminator, labels, y_fake, outputs_fake, criterion):
	y_gt = outputs_fake*0
	for i in range(len(y_gt)): y_gt[i] = criterion(labels[i],y_fake[i])
	y_gt = 2*this_discriminator.sg(y_gt)-1 # activation, to make it into [0,1] range
	return y_gt

def adhoc_normalization(outputs):
	outputs = (outputs-torch.mean(outputs,dim=1,keepdim=True))/torch.max(torch.abs(outputs),dim=1,keepdim=True).values
	return outputs

def DEBUG0001_training_sr_small_cnn(DEBUG_TRAINING_LOOP,x, this_net, labels, y_fake, outputs_fake ,y_gt):
	if DEBUG_TRAINING_LOOP: 
		this_net.forward_debug(x);
		print("  y_fake=%s,\n  labels=%s"%(str(y_fake),str(labels)))
		print("  y_fake.shape=%s, labels.shape=%s"%(str(y_fake.shape),str(labels.shape)))
		print("  outputs_fake:%s"%(str(outputs_fake)))
		print("  y_gt:%s"%(str(y_gt)))
		DEBUG_TRAINING_LOOP_signal = 1
	else:
		DEBUG_TRAINING_LOOP_signal = 0
	return DEBUG_TRAINING_LOOP_signal

def DEBUG0002_training_sr_small_cnn(DEBUG_TRAINING_LOOP, this_discriminator, xD, labels):
	DEBUG_TRAINING_LOOP_signal = 0
	if DEBUG_TRAINING_LOOP:
		outputs = this_discriminator.forward_debug(xD);
		# outputs = adhoc_normalization(outputs)
		y_gt = torch.ones(outputs.shape[0]).to(this_device)
		print("  [tr1] labels.shape:%s"%(str(labels.shape)))
		print("  [tr2] outputs:%s outputs.shape:%s"%(str(outputs),str(outputs.shape)))
		print("  [tr3] y_gt:%s y_gt.shape=%s"%(str(y_gt),str(y_gt.shape)))
		DEBUG_TRAINING_LOOP_signal = 1
	return DEBUG_TRAINING_LOOP_signal

def DEBUG0003_training_sr_small_cnn(DEBUG_TRAINING_LOOP, this_net, x, labels, this_discriminator):
	DEBUG_TRAINING_LOOP_signal = 0
	if DEBUG_TRAINING_LOOP:
		y_ind = torch.argmax(this_net(x).contiguous(), dim=1)
		y = np.zeros(shape=labels.shape)
		for j in range(len(y)): y[j][y_ind[j]] = 1
		y = torch.tensor(y).to(device=this_device).to(torch.float)
		print("  [tg 1] y:%s y.shape:%s"%(str(y),str(y.shape)))
		yD = this_discriminator.forward_debug((x,y)).contiguous()
		DEBUG_TRAINING_LOOP_signal = 1
	return DEBUG_TRAINING_LOOP_signal