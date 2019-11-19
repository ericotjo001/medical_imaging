from utils.utils import *

class SoftDiceLoss(nn.Module):
	"""
	sdl = SoftDiceLoss()
	sdl.smooth=1.
	logits = torch.Tensor([[1., 1., 0.],[1., 1., 0.]])
	targets = torch.Tensor([[1., 1., 0.],[1., 1., 0.]])
	targets2 = torch.Tensor([[0., 1., 0.],[1., 1., 0.]])
	targets3 = torch.Tensor([[0., 1., 1.],[1., 1., 0.]])
	targets4 = torch.Tensor([[0., 0., 1.],[0., 0., 1.]])
	t = [targets, targets2, targets3, targets4]
	for tt in t:
	    dice = sdl(logits, tt, factor=1).item()
	    print(dice)

	targetsALLZERO = torch.Tensor([[0., 0., 0.],[0., 0., 0.]])
	dice = sdl(logits, targetsALLZERO, factor=1).item()
	print("dice target all zero:", dice)
	dice = sdl(targetsALLZERO, targetsALLZERO, factor=1).item()
	print("dice both zeros", dice)

	Output:
	-1.0
	-0.875
	-0.7777777910232544
	-0.1428571492433548
	dice target all zero: -0.20000000298023224
	dice both zeros -1.0
	"""

	def __init__(self):
		super(SoftDiceLoss, self).__init__()
		self.smooth = 1

	def forward(self, logits, targets, factor=1, print_info=None):        
		num = targets.numel()
		m1 = logits.contiguous().view(-1).to(torch.float)
		m2 = targets.contiguous().view(-1).to(torch.float)
		# print(" .SoftDiceLoss().",m1.shape,m2.shape)
		if not m1.shape == m2.shape:
		   	raise Exception('SoftDiceLoss. forward(). Shapes do not match.') 
		intersection = m1 * m2
		if print_info is not None:
			if print_info >= 19:
				print("   m1.sum(), m2.sum()       = %d, %d\n   intersection.sum(), num =  %d, %d" % (m1.sum(), m2.sum(), intersection.sum(), num))

		score = (2. * intersection.sum() + self.smooth) / (m1.sum() + m2.sum() + self.smooth)
		score = (1 - score) * factor

		'''DEBUG'''
		# score = torch.Tensor(np.random.normal(0.5,1, size=(1,)))
		return score


def training_UNet3D_compute_loss(criterion, y, y0, config_data, this_net, x):
	DEBUG_SIGNAL_LOSS = False
	if DEBUG_TRAINING_LOOP_LOSS: 
		print("  utils.loss.py. training_UNet3D_compute_loss()")
		DEBUG_SIGNAL_LOSS = True

	if config_data['training_mode'] == 'UNet3D': loss = criterion(y, y0)
	elif config_data['training_mode'] == 'UNet3Db': loss = criterion(y, y0)
	elif config_data['training_mode'] == 'UNet3D_LRP_optim':
		if DEBUG_TRAINING_LOOP_LOSS: 
			LRP_optim_loss_DEBUG(this_net, x, y0, criterion)
			return None, DEBUG_SIGNAL_LOSS
		loss = criterion(y, y0) + LRP_optim_loss(this_net, x, y0, criterion)
	elif config_data['training_mode'] == 'UNet3Db_LRP_optim':
		regularization = 0.01
		loss = criterion(y, y0) + regularization * LRP_optim_loss(this_net, x, y0, criterion)
	return loss, DEBUG_SIGNAL_LOSS


def LRP_optim_loss(this_net, x_batch, y0_batch, criterion, threshold=0.5):
	lrp_loss = torch.tensor(0.).to(device=x_batch.device)

	R_batch = this_net(x_batch).contiguous()
	R_batch = this_net.relprop_skip(R_batch)
	this_net(x_batch, save_for_relprop=False).contiguous() #release memory for model saving
	# relprop for batch does not quite make sense with this implementation
	# assume batch_size is 1 for safety
	i_batch=0
	for i_batch in range(len(R_batch)):
		R = R_batch[i_batch]
		y0 = y0_batch[i_batch]
		sy0 = y0.shape
		for R_channel in R:
			R_channel = torch.stack((torch.zeros(R_channel.shape).to(device=R.device),R_channel),dim=0)
			R_channel = R_channel.reshape((1,)+R_channel.shape)
			y0 = y0.reshape((1,)+sy0)
			channel_lrp_loss = criterion(R_channel,y0)
			lrp_loss = lrp_loss +  channel_lrp_loss
	n_batch = i_batch + 1
	lrp_loss = lrp_loss/n_batch
	return lrp_loss

def LRP_optim_loss_DEBUG(this_net, x_batch, y0_batch, criterion, threshold=0.5):
	print("    LRP_optim_loss_DEBUG().")

	lrp_loss = torch.tensor(0.).to(device=x_batch.device)
	R_batch = this_net(x_batch).contiguous()
	R_batch = this_net.relprop_skip(R_batch)
	this_net(x_batch, save_for_relprop=False).contiguous() #release memory for model saving
	# relprop for batch does not quite make sense with this implementation
	# assume batch_size is 1 for safety
	i_batch=0
	for i_batch in range(len(R_batch)):
		R = R_batch[i_batch]
		y0 = y0_batch[i_batch]
		sy0 = y0.shape
		for R_channel in R:
			R_channel = torch.stack((torch.zeros(R_channel.shape).to(device=R.device),R_channel),dim=0)
			R_channel = R_channel.reshape((1,)+R_channel.shape)
			y0 = y0.reshape((1,)+sy0)
			print('      R_channel.shape:%s'%(str(R_channel.shape)))
			print('      y0.shape       :%s'%(str(y0.shape)))
			channel_lrp_loss = criterion(R_channel,y0)
			lrp_loss = lrp_loss +  channel_lrp_loss
			return

	n_batch = i_batch + 1
	lrp_loss = lrp_loss/n_batch
	return lrp_loss
