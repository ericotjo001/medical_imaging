from models.networks_components import * 

FCN8like_DEBUG = 0
FCN_DEBUG = 0
UNet3D_DEBUG = 1

class UNet3D(ModulePlus):
	def __init__(self,  device=None ):
		super(UNet3D, self).__init__()
		batch_norm = True
		input_channel = 6
		if not UNet3D_DEBUG:
			label = 1
			self.cblocks1 = ConvBlocksUNet(label, batch_norm, device)
			# in_channels, out_channels, kernel_sizes, paddings, strides, dilations
			self.cblocks1.conv_three_blocks([input_channel,32],[32,64],[3,3],[1,1],[1,1],[1,1])
			self.cblocks2 = ConvBlocksUNet(2, batch_norm, device)
			self.cblocks2.conv_three_blocks([64,64],[64,128],[3,3],[1,1],[1,1],[1,1])
			self.cblocks3 = ConvBlocksUNet(3, batch_norm, device)
			self.cblocks3.conv_three_blocks([128,128],[128,256],[3,3],[1,1],[1,1],[1,1])
			self.cblocks4 = ConvBlocksUNet(4, batch_norm, device)
			self.cblocks4.conv_three_blocks([256,256],[256,512],[3,3],[1,1],[1,1],[1,1])
			self.deconv1 = nn.ConvTranspose3d(512,512, 2, stride=2, bias=False).to(device=device)

			# RHS of the U shape
			self.cblocks5 = ConvBlocksUNet('3b', batch_norm, device)
			self.cblocks5.conv_three_blocks([256+512,256],[256,256],[3,3],[1,1],[1,1],[1,1])
			self.deconv2 = nn.ConvTranspose3d(256,256, 2, stride=2, bias=False).to(device=device)
			self.cblocks6 = ConvBlocksUNet('2b', batch_norm, device)
			self.cblocks6.conv_three_blocks([128+256,128],[128,128],[3,3],[1,1],[1,1],[1,1])
			self.deconv3 = nn.ConvTranspose3d(128,128, 2, stride=2, bias=False).to(device=device)
			self.cblocks7 = ConvBlocksUNet('1b', batch_norm, device)
			self.cblocks7.conv_three_blocks([64+128,64],[64,64],[3,3],[1,1],[1,1],[1,1])
			
			self.convf = nn.Conv3d(64, number_of_classes, 1, padding=0, stride=1, dilation=1).to(device=device)  
		else:
			self.cblocks1 = ConvBlocksUNet(1, batch_norm, device)
			self.cblocks1.conv_three_blocks([input_channel,3],[3,4],[3,3],[1,1],[1,1],[1,1])
			self.cblocks2 = ConvBlocksUNet(2, batch_norm, device)
			self.cblocks2.conv_three_blocks([4,4],[4,6],[3,3],[1,1],[1,1],[1,1])
			self.cblocks3 = ConvBlocksUNet(3, batch_norm, device)
			self.cblocks3.conv_three_blocks([6,6],[6,8],[3,3],[1,1],[1,1],[1,1])
			self.cblocks4 = ConvBlocksUNet(4, batch_norm, device)
			self.cblocks4.conv_three_blocks([8,8],[8,16],[3,3],[1,1],[1,1],[1,1])
			self.deconv1 = nn.ConvTranspose3d(16,16, 2, stride=2, bias=False).to(device=device)

			# RHS of the U shape
			self.cblocks5 = ConvBlocksUNet('3b', batch_norm, device)
			self.cblocks5.conv_three_blocks([8+16,8],[8,8],[3,3],[1,1],[1,1],[1,1])
			self.deconv2 = nn.ConvTranspose3d(8,8, 2, stride=2, bias=False).to(device=device)
			self.cblocks6 = ConvBlocksUNet('2b', batch_norm, device)
			self.cblocks6.conv_three_blocks([6+8,6],[6,6],[3,3],[1,1],[1,1],[1,1])
			self.deconv3 = nn.ConvTranspose3d(6,6, 2, stride=2, bias=False).to(device=device)
			self.cblocks7 = ConvBlocksUNet('1b', batch_norm, device)
			self.cblocks7.conv_three_blocks([4+6,4],[4,4],[3,3],[1,1],[1,1],[1,1])
			
			self.convf = nn.Conv3d(4, number_of_classes, 1, padding=0, stride=1, dilation=1).to(device=device)  

		self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True).to(device=device)
		self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True).to(device=device)
		self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True).to(device=device)

		self._init_weight()

	def forward(self, x):
		x = self.cblocks1(x)
		h1 = torch.Tensor(np.zeros(shape=(x.shape)))
		h1.data = x.clone()
		x = self.pool1(x)
		x = self.cblocks2(x)
		h2 = torch.Tensor(np.zeros(shape=(x.shape)))
		h2.data = x.clone()
		x = self.pool2(x)
		x = self.cblocks3(x)
		h3 = torch.Tensor(np.zeros(shape=(x.shape)))
		h3.data = x.clone()
		x = self.pool3(x)
		x = self.cblocks4(x)
		x = self.deconv1(x)[:,:,:-1,:,:] # the index is for "negative padding"

		x = torch.cat((h3,x),dim=1)	
		x = self.cblocks5(x)
		x = self.deconv2(x)
		x = torch.cat((h2,x),dim=1)	
		x = self.cblocks6(x)
		x = self.deconv3(x)[:,:,:-1,:,:] # the index is for "negative padding"
		x = torch.cat((h1,x),dim=1)
		x = self.cblocks7(x)

		x = self.convf(x)
		return x

	def forward_debug(self, x):
		print("UNet3D. forward_debug()")
		x = self.cblocks1(x)
		h1 = torch.Tensor(np.zeros(shape=(x.shape)))
		h1.data = x.clone()
		print("  [0] x.shape:%s, h1.shape:%s"%(str(x.shape), str(h1.shape)))
		x = self.pool1(x)

		x = self.cblocks2(x)
		h2 = torch.Tensor(np.zeros(shape=(x.shape)))
		h2.data = x.clone()
		print("  [1] x.shape:%s, h2.shape:%s"%(str(x.shape), str(h2.shape)))
		x = self.pool2(x)

		x = self.cblocks3(x)
		h3 = torch.Tensor(np.zeros(shape=(x.shape)))
		h3.data = x.clone()
		print("  [2] x.shape:%s, h3.shape:%s"%(str(x.shape), str(h3.shape)))
		x = self.pool3(x)

		x = self.cblocks4(x)
		print("  [3] x.shape:%s"%(str(x.shape)))

		x = self.deconv1(x)[:,:,:-1,:,:] # the index is for "negative padding"
		print("  [4] x.shape:%s (3D shape must be the same as [2])"%(str(x.shape)))
		x = torch.cat((h3,x),dim=1)
		print("    [4b] x.shape:%s (after concat)"%(str(x.shape)))
		
		x = self.cblocks5(x)
		x = self.deconv2(x)
		print("  [5] x.shape:%s (3D shape must be the same as [1])"%(str(x.shape)))
		x = torch.cat((h2,x),dim=1)
		print("    [5b] x.shape:%s (after concat)"%(str(x.shape)))
		
		x = self.cblocks6(x)
		x = self.deconv3(x)[:,:,:-1,:,:] # the index is for "negative padding"
		print("  [6] x.shape:%s (3D shape must be the same as [0])"%(str(x.shape)))
		x = torch.cat((h1,x),dim=1)
		print("    [6b] x.shape:%s (after concat)"%(str(x.shape)))

		x = self.cblocks7(x)
		print("  [7] x.shape:%s"%(str(x.shape)))

		x = self.convf(x)
		print("  output shape:%s"%(str(x.shape)))
		return x

class FCN8like(ModulePlus):
	"""docstring for FCN8like"""
	def __init__(self, device=None):
		super(FCN8like, self).__init__()
		
		if not FCN8like_DEBUG:
			# conv1 to fc7
			USE_LITE = 1
			if not USE_LITE:
	 			inc = [6 ,64,64 ,128,128,256, 256, 256,512, 512,512,512, 512,512, 4096 ] # in channels
	 			ouc = [64,64,128,128,256,256, 256, 512,512, 512,512,512, 512,4096,4096 ] # out channels
			else:
				inc = [6 ,32,32, 64, 64, 128, 128, 128,256, 256,256,512, 512, 512,2048 ] # in channels
				ouc = [32,32,64, 64, 128,128, 128, 256,256, 256,512,512, 512,2048,2048 ] # out channels

		else:
			''' debugging here '''
			inc = [6 ,3 , 3 ,  2,  2,  2,   2,  2,  2,  3,  3,  3,  4,  4,  4] # in channels
			ouc = [3 ,3 , 2 ,  2,  2,  2,   2,  2,  3,  3,  3,  4,  4,  4,  5] # out channels
		# conv1
		self.conv1_1 = nn.Conv3d(inc[0], ouc[0], 3, padding=(5,50,50)).to(device=device)
		self.relu1_1 = nn.ReLU(inplace=True).to(device=device)
		self.conv1_2 = nn.Conv3d(inc[1], ouc[1], 3, padding=1).to(device=device)
		self.relu1_2 = nn.ReLU(inplace=True).to(device=device)
		self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True).to(device=device)  # 1/2

		  # conv2
		self.conv2_1 = nn.Conv3d(inc[2],ouc[2], 3, padding=1).to(device=device)
		self.relu2_1 = nn.ReLU(inplace=True).to(device=device)
		self.conv2_2 = nn.Conv3d(inc[3],ouc[3], 3, padding=1).to(device=device)
		self.relu2_2 = nn.ReLU(inplace=True).to(device=device)
		self.pool2 = nn.MaxPool3d((1,2,2), stride=(1,2,2), ceil_mode=True).to(device=device)  # 1/4

		# conv3
		self.conv3_1 = nn.Conv3d(inc[4],ouc[4], 3, padding=1).to(device=device)
		self.relu3_1 = nn.ReLU(inplace=True).to(device=device)
		self.conv3_2 = nn.Conv3d(inc[5],ouc[5], 3, padding=1).to(device=device)
		self.relu3_2 = nn.ReLU(inplace=True).to(device=device)
		self.conv3_3 = nn.Conv3d(inc[6],ouc[6], 3, padding=1).to(device=device)
		self.relu3_3 = nn.ReLU(inplace=True).to(device=device)
		self.pool3 = nn.MaxPool3d((1,2,2), stride=(1,2,2), ceil_mode=True).to(device=device)  # 1/8

		# conv4
		self.conv4_1 = nn.Conv3d(inc[7],ouc[7], 3, padding=1).to(device=device)
		self.relu4_1 = nn.ReLU(inplace=True).to(device=device)
		self.conv4_2 = nn.Conv3d(inc[8],ouc[8], 3, padding=1).to(device=device)
		self.relu4_2 = nn.ReLU(inplace=True).to(device=device)
		self.conv4_3 = nn.Conv3d(inc[9],ouc[9], 3, padding=1).to(device=device)
		self.relu4_3 = nn.ReLU(inplace=True).to(device=device)
		self.pool4 = nn.MaxPool3d((1,2,2), stride=(1,2,2), ceil_mode=True)  # 1/16

		# conv5
		self.conv5_1 = nn.Conv3d(inc[10],ouc[10], 3, padding=1).to(device=device)
		self.relu5_1 = nn.ReLU(inplace=True).to(device=device)
		self.conv5_2 = nn.Conv3d(inc[11],ouc[11], 3, padding=1).to(device=device)
		self.relu5_2 = nn.ReLU(inplace=True).to(device=device)
		self.conv5_3 = nn.Conv3d(inc[12],ouc[12], 3, padding=1).to(device=device)
		self.relu5_3 = nn.ReLU(inplace=True).to(device=device)
		self.pool5 = nn.MaxPool3d((1,2,2), stride=(1,2,2), ceil_mode=True).to(device=device)  # 1/32

		# fc6
		self.fc6 = nn.Conv3d(inc[13],ouc[13], (2,7,7)).to(device=device)
		self.relu6 = nn.ReLU(inplace=True).to(device=device)
		self.drop6 = nn.Dropout3d().to(device=device)

		# fc7
		self.fc7 = nn.Conv3d(inc[14],ouc[14], 1).to(device=device)
		self.relu7 = nn.ReLU(inplace=True).to(device=device)
		self.drop7 = nn.Dropout3d().to(device=device)

		self.score_fr = nn.Conv3d(ouc[14], number_of_classes, 1).to(device=device)
		self.score_pool3 = nn.Conv3d(ouc[6], number_of_classes, 1).to(device=device)
		self.score_pool4 = nn.Conv3d(ouc[9], number_of_classes, 1).to(device=device)

		self.upscore2 = nn.ConvTranspose3d(number_of_classes, number_of_classes, (1,4,4), stride=(1,2,2), bias=False).to(device=device)
		self.upscore8 = nn.ConvTranspose3d(number_of_classes, number_of_classes, (3,24,24), stride=(2,12,12), bias=False).to(device=device)
		self.upscore_pool4 = nn.ConvTranspose3d(number_of_classes, number_of_classes, (1,4,4), stride=(1,2,2), bias=False).to(device=device)

		self._init_weight()
		
	def forward(self,x):
		h = x
		h = self.relu1_1(self.conv1_1(h))
		h = self.relu1_2(self.conv1_2(h))
		h = self.pool1(h)

		h = self.relu2_1(self.conv2_1(h))
		h = self.relu2_2(self.conv2_2(h))
		h = self.pool2(h)

		h = self.relu3_1(self.conv3_1(h))
		h = self.relu3_2(self.conv3_2(h))
		h = self.relu3_3(self.conv3_3(h))
		h = self.pool3(h)
		pool3 = h  # 1/8
		h = self.relu4_1(self.conv4_1(h))
		h = self.relu4_2(self.conv4_2(h))
		h = self.relu4_3(self.conv4_3(h))
		h = self.pool4(h)
		pool4 = h  # 1/16
		h = self.relu5_1(self.conv5_1(h))
		h = self.relu5_2(self.conv5_2(h))
		h = self.relu5_3(self.conv5_3(h))
		h = self.pool5(h)
		h = self.relu6(self.fc6(h))
		h = self.drop6(h)
		h = self.relu7(self.fc7(h))
		h = self.drop7(h)
		h = self.score_fr(h)
		h = self.upscore2(h)
		upscore2 = h  # 1/16
		factor1 = 1
		h = self.score_pool4(pool4*factor1)
		hs, us2 = h.shape, upscore2.shape
		D_start,H_start,W_start = int(np.floor(0.5*(hs[2]-us2[2]))), int(np.floor(0.5*(hs[3]-us2[3]))), int(np.floor(0.5*(hs[4]-us2[4]))) 
		h = h[:,:,D_start:D_start+us2[2],
			H_start:H_start+us2[3],
			W_start:W_start+us2[4]]
		score_pool4c = h			
		h = upscore2 + score_pool4c  # 1/16
		h = self.upscore_pool4(h)
		upscore_pool4 = h  # 1/8
		factor2 = 1
		h = self.score_pool3(pool3 * factor1)  
		hs, usp4s = h.shape, upscore_pool4.shape
		D_start,H_start,W_start = int(np.floor(0.5*(hs[2]-usp4s[2]))), int(np.floor(0.5*(hs[3]-usp4s[3]))), int(np.floor(0.5*(hs[4]-usp4s[4]))) 
		h = h[:,:,D_start:D_start+usp4s[2],
			H_start:H_start+usp4s[3],
			W_start:W_start+usp4s[4]]
		score_pool3c = h  # 1/8
		h = upscore_pool4 + score_pool3c  # 1/8
		h = self.upscore8(h)
		hs, xs = h.shape, x.shape
		D_start,H_start,W_start = int(np.floor(0.5*(hs[2]-xs[2]))), int(np.floor(0.5*(hs[3]-xs[3]))), int(np.floor(0.5*(hs[4]-xs[4]))) 
		h = h[:,:,D_start:D_start+xs[2],
			H_start:H_start+xs[3],
			W_start:W_start+xs[4]]
		return h

	def forward_debug(self, x):
		print("\nFCN8like forward_debug().")
		h = x
		print("  [-1] x.shape:%s"%(str(x.shape)))
		h = self.relu1_1(self.conv1_1(h))
		h = self.relu1_2(self.conv1_2(h))
		h = self.pool1(h)

		h = self.relu2_1(self.conv2_1(h))
		h = self.relu2_2(self.conv2_2(h))
		h = self.pool2(h)

		h = self.relu3_1(self.conv3_1(h))
		h = self.relu3_2(self.conv3_2(h))
		h = self.relu3_3(self.conv3_3(h))
		h = self.pool3(h)
		pool3 = h  # 1/8
		print("  [0] pool3.shape:%s"%(str(pool3.shape)))

		h = self.relu4_1(self.conv4_1(h))
		h = self.relu4_2(self.conv4_2(h))
		h = self.relu4_3(self.conv4_3(h))
		h = self.pool4(h)
		pool4 = h  # 1/16
		print("  [1] pool4.shape:%s"%(str(pool4.shape)))

		h = self.relu5_1(self.conv5_1(h))
		h = self.relu5_2(self.conv5_2(h))
		h = self.relu5_3(self.conv5_3(h))
		h = self.pool5(h)
		print("  [2] h.shape:%s"%(str(h.shape)))

		h = self.relu6(self.fc6(h))
		h = self.drop6(h)
		print("  [3] h.shape:%s"%(str(h.shape)))

		h = self.relu7(self.fc7(h))
		h = self.drop7(h)
		print("  [4] h.shape:%s"%(str(h.shape)))

		h = self.score_fr(h)
		h = self.upscore2(h)
		upscore2 = h  # 1/16
		print("  [5] upscore2.shape:%s"%(str(upscore2.shape)))

		'''
		Fusion part 1
		for cropping layer. WE simplify by always thaking the center of the crop.
		'''
		factor1 = 1

		h = self.score_pool4(pool4*factor1)
		hs, us2 = h.shape, upscore2.shape
		D_start,H_start,W_start = int(np.floor(0.5*(hs[2]-us2[2]))), int(np.floor(0.5*(hs[3]-us2[3]))), int(np.floor(0.5*(hs[4]-us2[4]))) 
		print("  [6a] pre score_pool4c.shape:%s"%(str(h.shape)))
		print("     D_start,H_start,W_start = %s, %s, %s"%(str(D_start),str(H_start),str(W_start)))
		h = h[:,:,D_start:D_start+us2[2],
			H_start:H_start+us2[3],
			W_start:W_start+us2[4]]
		score_pool4c = h
		print("  [6b] score_pool4c.shape:%s (must be the same as 5)"%(str(score_pool4c.shape)))
					
		h = upscore2 + score_pool4c  # 1/16
		h = self.upscore_pool4(h)
		upscore_pool4 = h  # 1/8
		print("  [7] upscore_pool4.shape:%s"%(str(upscore_pool4.shape)))


		'''
		Fusion part 2
		for cropping layer. WE simplify by always thaking the center of the crop.
		'''
		factor2 = 1
		h = self.score_pool3(pool3 * factor1)  
		hs, usp4s = h.shape, upscore_pool4.shape
		D_start,H_start,W_start = int(np.floor(0.5*(hs[2]-usp4s[2]))), int(np.floor(0.5*(hs[3]-usp4s[3]))), int(np.floor(0.5*(hs[4]-usp4s[4]))) 
		print("  [8a] score_pool3.shape:%s"%(str(h.shape)))
		print("     D_start,H_start,W_start = %s, %s, %s"%(str(D_start),str(H_start),str(W_start)))
		h = h[:,:,D_start:D_start+usp4s[2],
			H_start:H_start+usp4s[3],
			W_start:W_start+usp4s[4]]
		score_pool3c = h  # 1/8
		print("  [8b] score_pool3c.shape:%s (must be the same as [7])"%(str(score_pool3c.shape)))
		h = upscore_pool4 + score_pool3c  # 1/8

		'''
		Final fusion
		'''
		print("  [9] h.shape:%s"%(str(h.shape)))
		h = self.upscore8(h)
		hs, xs = h.shape, x.shape
		D_start,H_start,W_start = int(np.floor(0.5*(hs[2]-xs[2]))), int(np.floor(0.5*(hs[3]-xs[3]))), int(np.floor(0.5*(hs[4]-xs[4]))) 
		print("  [10] upscore8.shape:%s"%(str(h.shape)))
		h = h[:,:,D_start:D_start+xs[2],
			H_start:H_start+xs[3],
			W_start:W_start+xs[4]]
		print("  [final] h.shape=%s"%(str(h.shape)))
		return h

class CNN_basic(ModulePlus):
	"""docstring for CNN_basic"""
	def __init__(self, device=None):
		super(CNN_basic, self).__init__()
		self.verbose = 0
		self.without_bn = True
		self.cblocks_dict = {}

		''' Edit here'''
		self.cblocks_dict['in_channels']  = [6,7,32,32,48,64]
		self.cblocks_dict['out_channels'] = [7,32,32,48,64,1]
		self.cblocks_dict['kernel_sizes'] = [5,5,7,7,9,9]
		self.cblocks_dict['paddings'] = [2,2,3,3,4,4]
		self.cblocks_dict['strides'] = [1,1,1,1,1,1] 
		self.cblocks_dict['dilations'] = [1,1,1,1,1,1]

		''' For testing'''
		# self.cblocks_dict['in_channels']  = [6,2,2,3]
		# self.cblocks_dict['out_channels'] = [2,2,3,1]
		# self.cblocks_dict['kernel_sizes'] = [3,3,3,3]
		# self.cblocks_dict['paddings'] = [1,1,1,1]
		# self.cblocks_dict['strides'] = [1,1,1,1] 
		# self.cblocks_dict['dilations'] = [1,1,1,1]

		self.cblocks = ConvBlocks(device=device).to(device=device)
		self.set_up_layers()
		self.convf = nn.ConvTranspose3d(1, number_of_classes, 1, padding=0, stride=1, dilation=1).to(device=device)

	def set_up_layers(self):
		if self.without_bn:
			self.cblocks.convblocks_without_bn(self.cblocks_dict['in_channels'], 
				self.cblocks_dict['out_channels'] , 
				self.cblocks_dict['kernel_sizes'], 
				self.cblocks_dict['paddings'], 
				self.cblocks_dict['strides'], 
				self.cblocks_dict['dilations'])
		else:
			self.cblocks.convblocks(self.cblocks_dict['in_channels'], 
				self.cblocks_dict['out_channels'] , 
				self.cblocks_dict['kernel_sizes'], 
				self.cblocks_dict['paddings'], 
				self.cblocks_dict['strides'], 
				self.cblocks_dict['dilations'])
		
	def forward(self, x):
		x = self.cblocks.forward(x)
		x = self.convf(x)
		return x

	def forward_debug(self, x):
		print("\nforward_debug().")
		print("  [-1] x.shape:",x.shape)
		x = self.cblocks.forward(x)
		x = self.convf(x)
		print("  [0] x.shape:",x.shape)
		return x


class FCN(ModulePlus):
	"""FCN: this appears to perform horribly"""
	def __init__(self, device=None):
		super(FCN, self).__init__()
		self.verbose = 0

		self.cblocks_dict = {}
		
		if not FCN_DEBUG:
			''' Edit here '''
			# For testing set     "resize": "[124,124,9]" in the config
			self.out_filter_conv = 128
			self.cblocks_dict['in_channels']  = [6,16,32,64,64,128]
			self.cblocks_dict['out_channels'] = [16,32,64,64,128,self.out_filter_conv]
			self.cblocks_dict['kernel_sizes'] = [3,3,3,3,3,3]
			self.cblocks_dict['paddings'] = [1,1,1,1,1,1]
			self.cblocks_dict['strides'] = [1,1,1,1,1,1] 
			self.cblocks_dict['dilations'] = [1,1,1,1,1,1]
		else:
			'''	For testing '''
			self.out_filter_conv = 3
			self.cblocks_dict['in_channels']  = [6,2,3,3]
			self.cblocks_dict['out_channels'] = [2,3,3,self.out_filter_conv]
			self.cblocks_dict['kernel_sizes'] = [3,3,3,3]
			self.cblocks_dict['paddings'] = [1,1,1,1]
			self.cblocks_dict['strides'] = [1,1,1,1] 
			self.cblocks_dict['dilations'] = [1,1,1,1]

		self.cblocks = ConvBlocksPool(device=device).to(device=device)
		self.set_up_layers()
	
	def set_up_layers(self):
		self.cblocks.convblocks_with_pool(self.cblocks_dict['in_channels'], 
			self.cblocks_dict['out_channels'] , 
			self.cblocks_dict['kernel_sizes'], 
			self.cblocks_dict['paddings'], 
			self.cblocks_dict['strides'], 
			self.cblocks_dict['dilations'])

		if not FCN_DEBUG:
			''' Edit here '''
			sl1 = [self.out_filter_conv,128] # input filter
			sl2 = [128,256] # output filter
			ks = [5,1] #kernel
			ps = [2,0] #padding
		else:
			'''	For testing '''
			sl1 = [self.out_filter_conv,3] # input filter
			sl2 = [3,3] # output filter
			ks = [5,1] #kernel
			ps = [2,0] #padding

		self.fc1 = nn.Conv3d(sl1[0], sl1[1], ks[0], padding=ps[0]).to(device=this_device); self.drop1 = nn.Dropout3d() # this fc is implemented by very large conv instead
		self.fc2 = nn.Conv3d(sl2[0], sl2[1], ks[1], padding=ps[1]).to(device=this_device); self.drop2 = nn.Dropout3d()
		self.score_fr = nn.Conv3d(sl2[1], 1, 1).to(device=this_device)

		if not FCN_DEBUG: self.upscore = nn.ConvTranspose3d(1,number_of_classes,(1,164,164), stride=(1,14,14), bias=False).to(device=this_device)
		else: self.upscore = nn.ConvTranspose3d(1,number_of_classes,(1,26,26), stride=(1,14,14), bias=False).to(device=this_device)

	def forward(self,x):
		x = self.cblocks.forward(x)
		x = self.drop1(F.elu(self.fc1(x)))
		x = self.drop2(F.elu(self.fc2(x)))
		x = self.score_fr(x)
		x = self.upscore(x)
		return x

	def forward_debug(self, x):
		print("\nforward_debug().")
		print("[-1] x.shape:",x.shape)
		x = self.cblocks.forward(x)
		print("[0] x.shape:",x.shape)
		x = self.drop1(F.elu(self.fc1(x)))
		print("[1] x.shape:",x.shape)
		x = self.drop2(F.elu(self.fc2(x)))
		print("[2] x.shape:",x.shape)
		x = self.score_fr(x)
		print("[3] x.shape:",x.shape)
		x = self.upscore(x)
		print("[4] x.shape:",x.shape)
		return x

def count_parameters(model, print_param=False):
	if print_param:
		for param in model.parameters(): print(param)
	num_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
	num_grad = sum(p.numel() for p in model.parameters())
	print("networks.py. count_parameters()\n  with grad: %s, with or without: %s"%(num_with_grad, num_grad))
	return num_with_grad, num_grad
