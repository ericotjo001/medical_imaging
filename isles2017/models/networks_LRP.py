from utils.utils import *

def relprop_size_adjustment(Z,R):
	sZ, sR = Z.shape, R.shape
	if not np.all(sZ==sR):
		tempR, tempZ = get_zero_container(Z,R), get_zero_container(Z,R)
		tempR[:sR[0],:sR[1],:sR[2],:sR[3],:sR[4]] = R; # R = tempR
		tempZ[:sZ[0],:sZ[1],:sZ[2],:sZ[3],:sZ[4]] = Z; # Z = tempZ
	else: return Z, R
	return tempZ, tempR

class MaxPool3dLRP(nn.MaxPool3d):
	"""docstring for MaxPool3dLRP"""
	def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
		super(MaxPool3dLRP, self).__init__(kernel_size, stride=stride, padding=padding, 
			dilation=1, return_indices=False, ceil_mode=ceil_mode)
		self.stride = stride
		self.padding = padding
		self.dilation = 1
		self.ceil_mode = ceil_mode
		self.buffer =1e-9 # 0.1 # 1e-9 # 0.1

	def forward(self,x):
		self.X = x
		return super(MaxPool3dLRP, self).forward(x)

	def gradprop(self, DY):
		DX = self.X * 0
		temp, indices = F.max_pool3d(self.X, self.kernel_size, self.stride,  self.padding, 
			self.dilation, self.ceil_mode, True)
		si, sDY = indices.shape, DY.shape
		if not np.all(si==sDY):
			tempi, tempDY = get_zero_container(indices,DY), get_zero_container(indices,DY) 
			tempi[:si[0],:si[1],:si[2],:si[3],:si[4]] = indices; indices = tempi
			tempDY[:sDY[0],:sDY[1],:sDY[2],:sDY[3],:sDY[4]] = DY; DY = tempDY
		DX = F.max_unpool3d(DY.to(device=indices.device), indices.to(torch.long), self.kernel_size, self.stride, self.padding)
		return DX

	def relprop(self,R, verbose=0):
		Z = self.forward(self.X) + self.buffer # 1e-9
		if verbose>99: print("  MaxPool3dLRP. relprop():\n    R.shape=%s,Z.shape=%s"%(str(R.shape),str(Z.shape)))
		Z, R = relprop_size_adjustment(Z,R)
		Z = Z + self.buffer
		S = R/Z
		C = self.gradprop(S)
		X, C = relprop_size_adjustment(self.X, C)
		R = X * C
		return R

class Conv3dLRP(nn.Conv3d):
	def __init__(self, in_channels, out_channels, kernel_size, 
		stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(Conv3dLRP, self).__init__(in_channels, out_channels, kernel_size, 
			stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
		self.Y = None
		self.X = None
		self.not_first_layer = True
		self.arg = (in_channels, out_channels, kernel_size, 
			stride, padding, dilation, groups, bias)

		self.relprop_max_min = None # typically [0,1]
		self.buffer = 1e-9 # 0.1 

	def gradprop(self, DY):
		# print("gradprop(). Reminder: adjust output_padding when necessary")
		return F.conv_transpose3d(DY.to(device=self.weight.device), self.weight, stride=self.stride, 
			padding=self.padding, output_padding=0)
	
	def forward(self,x):
		self.X=x
		return super(Conv3dLRP, self).forward(x)

	def relprop(self, R, verbose=0):
		in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = self.arg
		
		pself = Conv3dLRP(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		pself.to(device=self.X.device)
		pself.bias = nn.Parameter(pself.bias* 0); 
		temp = np.maximum(0,pself.weight.detach().cpu().numpy())
		pself.W = nn.Parameter(torch.tensor(temp))
		
		if self.not_first_layer:	
			
			# MEMORY NOT ENOUGH
			Z = super(Conv3dLRP, pself).forward(self.X) 
			if verbose>99: print("  Conv3dLRP. relprop():\n    R.shape=%s,Z.shape=%s"%(str(R.shape),str(Z.shape)))
			X = self.X.clone()
			Z, R = relprop_size_adjustment(Z,R)
			Z = Z + self.buffer
			if verbose>249: 
				print("    np.max(Z)=%s, np.min(Z)=%s"%(str(np.max(Z.view(-1).detach().cpu().numpy())),\
					str(np.min(Z.view(-1).detach().cpu().numpy()))))
				print("    np.max(R)=%s, np.min(R)=%s"%(str(np.max(R.view(-1).detach().cpu().numpy())),\
					str(np.min(R.view(-1).detach().cpu().numpy()))))

			R,Z =R.cpu(), Z.cpu()
			S = R / Z
			C = pself.gradprop(S)
			X, C = relprop_size_adjustment(X, C)
			R = X * C 
			
			if verbose>249: 
				print("    np.max(S)=%s, np.min(S)=%s"%(str(np.max(S.view(-1).detach().cpu().numpy())),\
					str(np.min(S.view(-1).detach().cpu().numpy()))))
				print("    np.max(C)=%s, np.min(C)=%s"%(str(np.max(C.view(-1).detach().cpu().numpy())),\
					str(np.min(C.view(-1).detach().cpu().numpy()))))
				print("    np.max(X)=%s, np.min(X)=%s"%(str(np.max(X.view(-1).detach().cpu().numpy())),\
					str(np.min(X.view(-1).detach().cpu().numpy()))))
		else:
			iself = Conv3dLRP(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
			iself.to(device=self.X.device)
			iself.bias = nn.Parameter(iself.bias* 0);

			nself = Conv3dLRP(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
			nself.to(device=self.X.device)
			nself.bias = nn.Parameter(nself.bias* 0); 
			tempn = np.minimum(0,nself.weight.detach().cpu().numpy())
			nself.W = nn.Parameter(torch.tensor(tempn))
			
			X, L, H = self.X.clone(), self.X*0 + self.relprop_max_min[0] , self.X*0 +self.relprop_max_min[1]

			Z = super(Conv3dLRP, iself).forward(X) - super(Conv3dLRP, pself).forward(L)\
			 - super(Conv3dLRP, nself).forward(H) 
			Z, R = relprop_size_adjustment(Z,R)
			Z = Z + self.buffer # 1e-9

			if verbose>99: print("  Conv3dLRP [FIRST LAYER]. relprop():\n    R.shape=%s,Z.shape=%s"%(str(R.shape),str(Z.shape)))
			S = R / Z

			iC,pC,nC = iself.gradprop(S), pself.gradprop(S), nself.gradprop(S)
			X, iC = relprop_size_adjustment(X, iC)
			L, pC = relprop_size_adjustment(L, pC)
			H, nC = relprop_size_adjustment(H, nC)
			R = X*iC-L*pC-H*nC
			if verbose>99: print("    shapes L,pC,H,nC=%s,%s,%s,%s"%(str(L.shape),str(pC.shape),str(H.shape),str(nC.shape),))
			
		# if verbose>99: print("    np.max(R)=%s, np.min(R)=%s"%(str(np.max(R.detach().cpu().numpy())),str(np.min(R.detach().cpu().numpy()))))
		return R

class ConvTranspose3dLRP(nn.ConvTranspose3d):
	def __init__(self, in_channels, out_channels, kernel_size, \
		stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
		super(ConvTranspose3dLRP, self).__init__(in_channels, out_channels, kernel_size, \
			stride=stride, padding=padding, output_padding=output_padding, groups=groups, \
			bias=bias, dilation=dilation, padding_mode=padding_mode)
		self.Y = None
		self.X = None
		self.not_first_layer = True
		self.arg = (in_channels, out_channels, kernel_size, 
			stride, padding, output_padding, groups, bias, dilation, padding_mode)
		self.buffer = 1e-9 # 0.1

	def forward(self,x):
		self.X =  x
		return super(ConvTranspose3dLRP, self).forward(x)
		
	def gradprop(self, DY):
		# print("ConvTranspose3dLRP.gradprop().")
		return F.conv3d(DY.to(device=self.weight.device), self.weight, bias=None, stride=self.stride, padding=self.padding, \
			dilation=self.dilation, groups=self.groups)

	def relprop(self, R, verbose=0):
		if self.not_first_layer:
			in_channels, out_channels, kernel_size, stride, padding, output_padding, \
				groups, bias, dilation, padding_mode = self.arg
			pself = ConvTranspose3dLRP(in_channels, out_channels, kernel_size, \
				stride, padding, output_padding, groups, bias, dilation, padding_mode)
			pself.to(device=self.X.device)
			
			if pself.bias is not None: pself.bias = nn.Parameter(pself.bias* 0); 
			temp = np.maximum(0,pself.weight.detach().cpu().numpy())
			pself.W = nn.Parameter(torch.tensor(temp))
			
			# MEMORY NOT ENOUGH
			Z = super(ConvTranspose3dLRP, pself).forward(self.X) # cpu() because cuda runs out of memory
			Z,R = relprop_size_adjustment(Z,R)
			Z = Z + self.buffer # 1e-9

			Z, R = Z.cpu(), R.cpu()
			S = R / Z
			C = pself.gradprop(S)

			X, C = relprop_size_adjustment(self.X, C)
			R = X * C
			
			return R.to(device=this_device)

		return None

class BatchNorm3dLRP(nn.BatchNorm3d):
	def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
		super(BatchNorm3dLRP, self).__init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

	def relprop(self, R):
		return R

class ReLU_LRP(nn.ReLU):
	def __init__(self, ):
		super(ReLU_LRP, self).__init__()
	def relprop(self,R): 
		return R

class LeakyReLU_LRP(nn.LeakyReLU):
	"""docstring for LeakyReLU"""
	def __init__(self,negative_slope=0.01, inplace=True):
		super(LeakyReLU_LRP, self).__init__(negative_slope=negative_slope,inplace=inplace)
	def relprop(self,R): 
		return R
		
						