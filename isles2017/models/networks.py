from models.networks_components import * 
import models.networks_LRP as lr


class UNet3D(ModulePlus):
	# no of param 19 077 158
	def __init__(self, no_of_input_channel=6, with_LRP=False, setup_lens_module=False):
		super(UNet3D, self).__init__()
		batch_norm = True
		self.first_layer_normalization = [0,1]
		self.no_of_input_channel = no_of_input_channel
		self.with_LRP = with_LRP
		
		label = 1
		if not UNet3D_DEBUG:			
			cb1 = [[self.no_of_input_channel,32],[32,64],[3,3],[1,1],[1,1],[1,1]]
			cb2 = [[64,64],[64,128],[3,3],[1,1],[1,1],[1,1]]
			cb3 = [[128,128],[128,256],[3,3],[1,1],[1,1],[1,1]]
			cb4 = [[256,256],[256,512],[3,3],[1,1],[1,1],[1,1]]
			
			dc1 = [512,512,2]
			cb5 = [[256+512,256],[256,256],[3,3],[1,1],[1,1],[1,1]]
			dc2 = [256,256,2]
			cb6 = [[128+256,128],[128,128],[3,3],[1,1],[1,1],[1,1]]
			dc3 = [128,128,2]
			cb7 = [[64+128,64],[64,64],[3,3],[1,1],[1,1],[1,1]]
			cf = [64]
						
		else:
			cb1 = [[self.no_of_input_channel,3],[3,4],[3,3],[1,1],[1,1],[1,1]]
			cb2 = [[4,4],[4,6],[3,3],[1,1],[1,1],[1,1]]
			cb3 = [[6,6],[6,8],[3,3],[1,1],[1,1],[1,1]]
			cb4 = [[8,8],[8,16],[3,3],[1,1],[1,1],[1,1]]

			dc1 = [16,16, 2]
			cb5 = [[8+16,8],[8,8],[3,3],[1,1],[1,1],[1,1]]
			dc2 = [8,8, 2]
			cb6 = [[6+8,6],[6,6],[3,3],[1,1],[1,1],[1,1]]
			dc3 = [6,6, 2]
			cb7 = [[4+6,4],[4,4],[3,3],[1,1],[1,1],[1,1]]
			cf = [4]
		
		self.cblocks1 = ConvBlocksUNet(label, batch_norm, with_LRP=self.with_LRP, is_the_first_block=True)
		self.cblocks1.relprop_max_min = self.first_layer_normalization
		self.cblocks1.conv_three_blocks(cb1[0],cb1[1],cb1[2], cb1[3],cb1[4],cb1[5],)
		self.pool1 = lr.MaxPool3dLRP(2, stride=2, ceil_mode=True)#.to(device=device)

		self.cblocks2 = ConvBlocksUNet(2, batch_norm, with_LRP=self.with_LRP)
		self.cblocks2.conv_three_blocks(cb2[0],cb2[1],cb2[2], cb2[3],cb2[4],cb2[5],)
		self.pool2 = lr.MaxPool3dLRP(2, stride=2, ceil_mode=True)#.to(device=device)

		self.cblocks3 = ConvBlocksUNet(3, batch_norm, with_LRP=self.with_LRP)
		self.cblocks3.conv_three_blocks(cb3[0],cb3[1],cb3[2], cb3[3],cb3[4],cb3[5],)
		self.pool3 = lr.MaxPool3dLRP(2, stride=2, ceil_mode=True)#.to(device=device)

		self.cblocks4 = ConvBlocksUNet(4, batch_norm, with_LRP=self.with_LRP)
		self.cblocks4.conv_three_blocks(cb4[0],cb4[1],cb4[2], cb4[3],cb4[4],cb4[5],)
		self.deconv1 = lr.ConvTranspose3dLRP(dc1[0],dc1[1], dc1[2], stride=2, bias=False)#.to(device=device)

		# RHS of the U shape
		self.cblocks5 = ConvBlocksUNet('3b', batch_norm, with_LRP=self.with_LRP)
		self.cblocks5.conv_three_blocks(cb5[0],cb5[1],cb5[2], cb5[3],cb5[4],cb5[5],)
		self.deconv2 = lr.ConvTranspose3dLRP(dc2[0], dc2[1], dc2[2], stride=2, bias=False)#.to(device=device)
		self.cblocks6 = ConvBlocksUNet('2b', batch_norm, with_LRP=self.with_LRP)
		self.cblocks6.conv_three_blocks(cb6[0],cb6[1],cb6[2], cb6[3],cb6[4],cb6[5],)
		self.deconv3 = lr.ConvTranspose3dLRP(dc3[0], dc3[1], dc3[2], stride=2, bias=False)#.to(device=device)
		self.cblocks7 = ConvBlocksUNet('1b', batch_norm, with_LRP=self.with_LRP)
		self.cblocks7.conv_three_blocks(cb7[0],cb7[1],cb7[2], cb7[3],cb7[4],cb7[5],)

		self.convf = lr.Conv3dLRP(cf[0], number_of_classes, 1, padding=0, stride=1, dilation=1)#.to(device=device)  
		self.bn = lr.BatchNorm3dLRP(number_of_classes)
		self.final_relu = lr.ReLU_LRP()


		for x in self.modules(): x = torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count()))
		self._init_weight()

		self.paramdict = { 
			'cb1': cb1, 'cb2': cb2, 'cb3': cb3, 'cb4': cb4, 'dc1': dc1,
			'cb5':cb5, 'dc2':dc2, 'cb6':cb6, 'dc3':dc3, 'cb7':cb7,
			'cf':cf,
			}
		# for x in self.paramdict:
		# 	print("    %s: %s"%(str(x),str(self.paramdict[x])))

	def forward(self, x, save_for_relprop=True):
		x = self.cblocks1(x, save_for_relprop=save_for_relprop)
		h1 = torch.Tensor(np.zeros(shape=(x.shape)))
		h1.data = x.clone()
		x = self.pool1(x, save_for_relprop=save_for_relprop)
		x = self.cblocks2(x, save_for_relprop=save_for_relprop)
		h2 = torch.Tensor(np.zeros(shape=(x.shape)))
		h2.data = x.clone()
		x = self.pool2(x, save_for_relprop=save_for_relprop)
		x = self.cblocks3(x, save_for_relprop=save_for_relprop)
		h3 = torch.Tensor(np.zeros(shape=(x.shape)))
		h3.data = x.clone()
		x = self.pool3(x, save_for_relprop=save_for_relprop)
		x = self.cblocks4(x, save_for_relprop=save_for_relprop)
		x = self.deconv1(x, save_for_relprop=save_for_relprop)[:,:,:-1,:,:] # the index is for "negative padding"

		x = torch.cat((h3,x),dim=1)	
		x = self.cblocks5(x, save_for_relprop=save_for_relprop)
		x = self.deconv2(x, save_for_relprop=save_for_relprop)
		x = torch.cat((h2,x),dim=1)	
		x = self.cblocks6(x, save_for_relprop=save_for_relprop)
		x = self.deconv3(x, save_for_relprop=save_for_relprop)[:,:,:-1,:,:] # the index is for "negative padding"
		x = torch.cat((h1,x),dim=1)
		x = self.cblocks7(x, save_for_relprop=save_for_relprop)

		x = self.convf(x, save_for_relprop=save_for_relprop)
		x = self.bn(x)
		x = self.final_relu(x) # F.relu(x)
		return x

	def forward_debug(self, x, save_for_relprop=True):
		print("UNet3D. forward_debug()")
		print("  [-1] x.shape:%s"%(str(x.shape)))
		x = self.cblocks1(x, save_for_relprop=save_for_relprop)
		h1 = torch.Tensor(np.zeros(shape=(x.shape)))
		h1.data = x.clone()
		print("  [0] x.shape:%s, h1.shape:%s"%(str(x.shape), str(h1.shape)))
		x = self.pool1(x, save_for_relprop=save_for_relprop)

		x = self.cblocks2(x, save_for_relprop=save_for_relprop)
		h2 = torch.Tensor(np.zeros(shape=(x.shape)))
		h2.data = x.clone()
		print("  [1] x.shape:%s, h2.shape:%s"%(str(x.shape), str(h2.shape)))
		x = self.pool2(x, save_for_relprop=save_for_relprop)

		x = self.cblocks3(x, save_for_relprop=save_for_relprop)
		h3 = torch.Tensor(np.zeros(shape=(x.shape)))
		h3.data = x.clone()
		print("  [2] x.shape:%s, h3.shape:%s"%(str(x.shape), str(h3.shape)))
		x = self.pool3(x, save_for_relprop=save_for_relprop)

		x = self.cblocks4(x, save_for_relprop=save_for_relprop)
		print("  [3] x.shape:%s"%(str(x.shape)))

		x = self.deconv1(x, save_for_relprop=save_for_relprop)[:,:,:-1,:,:] # the index is for "negative padding"
		print("  [4] x.shape:%s (3D shape must be the same as [2])"%(str(x.shape)))
		x = torch.cat((h3,x),dim=1)
		print("    [4b] x.shape:%s (after concat)"%(str(x.shape)))
		
		x = self.cblocks5(x, save_for_relprop=save_for_relprop)
		x = self.deconv2(x, save_for_relprop=save_for_relprop)
		print("  [5] x.shape:%s (3D shape must be the same as [1])"%(str(x.shape)))
		x = torch.cat((h2,x),dim=1)
		print("    [5b] x.shape:%s (after concat)"%(str(x.shape)))
		
		x = self.cblocks6(x, save_for_relprop=save_for_relprop)
		x = self.deconv3(x, save_for_relprop=save_for_relprop)[:,:,:-1,:,:] # the index is for "negative padding"
		print("  [6] x.shape:%s (3D shape must be the same as [0])"%(str(x.shape)))
		x = torch.cat((h1,x),dim=1)
		print("    [6b] x.shape:%s (after concat)"%(str(x.shape)))

		x = self.cblocks7(x, save_for_relprop=save_for_relprop)
		print("  [7] x.shape:%s"%(str(x.shape)))

		x = self.convf(x, save_for_relprop=save_for_relprop)
		x = self.bn(x)
		x = self.final_relu(x) # F.relu(x)
		print("  output shape:%s"%(str(x.shape)))
		return x

	def relnormalize(self, R, relprop_config, verbose=0):
		if relprop_config['normalization'] == 'raw':
			ss = 1.
		elif relprop_config['normalization'] == 'standard':
			# this is bad. the sum can go too large
			ss = torch.sum(R**2)**0.5
			R = R/ss
		elif relprop_config['normalization'] == 'fraction_pass_filter':
			ss = torch.max(torch.FloatTensor.abs(R))
			R = R/ss
			ppf = relprop_config['fraction_pass_filter']['positive']
			npf = relprop_config['fraction_pass_filter']['negative']
			Rplus = R*(R>=ppf[0]).to(torch.float)*(R<=ppf[1]).to(torch.float)
			Rmin = R*(R>=npf[0]).to(torch.float)*(R<=npf[1]).to(torch.float)
			R = ss*(Rmin+Rplus)
		elif relprop_config['normalization'] == 'fraction_clamp_filter':
			ss = torch.max(torch.FloatTensor.abs(R))
			R = R/ss
			ppf = relprop_config['fraction_clamp_filter']['positive']
			npf = relprop_config['fraction_clamp_filter']['negative']
			Rplus = torch.clamp(R,min=ppf[0], max=ppf[1])*(R>=0).to(torch.float)	
			Rmin = 	torch.clamp(R,min=npf[0], max=npf[1])*(R<0).to(torch.float)
			R = ss*(Rmin+Rplus)
		else: raise Exception('Invalid mode')
		return R, ss

	def relprop(self,R,relprop_config):
		if relprop_config['mode'] == 'UNet3D_standard': 
			R = self.relprop_standard(R, relprop_config)
		else: raise Exception('Invalid mode')
		return R

	def relprop_standard(self,R , relprop_config):
		R = self.final_relu.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		R = self.bn.relprop(R)
		R = self.convf.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		
		R = self.cblocks7.relprop(R) 
		R = R[:,self.paramdict['cb1'][1][1]:,:,:,:]; R, ss = self.relnormalize(R, relprop_config)
		h1 = R[:,:self.paramdict['cb1'][1][1],:,:,:] ; h1, ss = self.relnormalize(h1, relprop_config)
		R = self.deconv3.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		
		R = self.cblocks6.relprop(R) ; 
		R = R[:,self.paramdict['cb2'][1][1]:]; R, ss = self.relnormalize(R, relprop_config)
		h2 = R[:,:self.paramdict['cb2'][1][1],:,:,:] ;  h2, ss = self.relnormalize(h2, relprop_config)
		R = self.deconv2.relprop(R) ; R, ss = self.relnormalize(R, relprop_config)
		
		R = self.cblocks5.relprop(R)
		R = R[:,self.paramdict['cb3'][1][1]:]; R, ss = self.relnormalize(R, relprop_config)
		h3 = R[:,:self.paramdict['cb3'][1][1],:,:,:] ; h3, ss = self.relnormalize(h3, relprop_config)
		R = self.deconv1.relprop(R) ; R, ss = self.relnormalize(R, relprop_config)

		R = self.cblocks4.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		R = self.pool3.relprop(R)

		factor = relprop_config['UNet3D']['concat_factors'][0]
		R = factor*h3 + (1-factor)*centre_crop_tensor(R, h3.shape).to(this_device)
		R = self.cblocks3.relprop(R,verbose=0); R, ss = self.relnormalize(R, relprop_config)
		R = self.pool2.relprop(R)

		factor2 = relprop_config['UNet3D']['concat_factors'][1]
		R = factor2*h2 + (1-factor2)* centre_crop_tensor(R, h2.shape).to(this_device)
		R = self.cblocks2.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		R = self.pool1.relprop(R)

		factor3 = relprop_config['UNet3D']['concat_factors'][2]
		R = factor3*h1 + (1-factor3)*centre_crop_tensor(R, h1.shape).to(this_device)
		R = self.cblocks1.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		return R

	def relprop_debug(self, R, relprop_config):
		if relprop_config['mode'] == 'UNet3D_standard': 
			R = self.relprop_standard_debug(R, relprop_config)
		else: raise Exception('Invalid mode')
		return R

	def relprop_standard_debug(self,R , relprop_config):
		# print("%-8s  |  %24s  |  %24s  |"%(str(""),str(""),str("")))
		print("%-8s| %-24s | %-24s |  %-24s | %8s"%(str(""),str("max"),str("max"),str("shape"),str("numel")))
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[-1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		"""
		center_crop is the shape of R to output, assuming that
		  the shape of R is initially larger than the intended shape
		"""
		DEBUG_INTERNAL = 1

		R = self.final_relu.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		R = self.bn.relprop(R)
		R = self.convf.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		
		R = self.cblocks7.relprop(R) 
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		R = R[:,self.paramdict['cb1'][1][1]:,:,:,:]; R, ss = self.relnormalize(R, relprop_config)
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[1.1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		h1 = R[:,:self.paramdict['cb1'][1][1],:,:,:] ; h1, ss = self.relnormalize(h1, relprop_config)
		if DEBUG_INTERNAL: print("%-8s| %24s | %24s |  %24s | %8s"%(str("[1.1.h1]"),str(torch.max(h1).item()),str(torch.min(h1).item()),str(list(h1.shape)),str(h1.numel())))
		R = self.deconv3.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[1.2]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		
		R = self.cblocks6.relprop(R) ; 
		if DEBUG_INTERNAL: print("%-8s| %24s | %24s |  %24s | %8s"%(str("[2]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		R = R[:,self.paramdict['cb2'][1][1]:]; R, ss = self.relnormalize(R, relprop_config)
		if DEBUG_INTERNAL: print("%-8s| %24s | %24s |  %24s | %8s"%(str("[2.1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		h2 = R[:,:self.paramdict['cb2'][1][1],:,:,:] ;  h2, ss = self.relnormalize(h2, relprop_config)
		if DEBUG_INTERNAL: print("%-8s| %24s | %24s |  %24s | %8s"%(str("[2.1.h2]"),str(torch.max(h2).item()),str(torch.min(h2).item()),str(list(h2.shape)),str(h2.numel())))
		R = self.deconv2.relprop(R) ; R, ss = self.relnormalize(R, relprop_config)
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[2.2]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
	

		R = self.cblocks5.relprop(R)
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[3]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		R = R[:,self.paramdict['cb3'][1][1]:]; R, ss = self.relnormalize(R, relprop_config)
		if DEBUG_INTERNAL: print("%-8s| %24s | %24s |  %24s | %8s"%(str("[3.1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		h3 = R[:,:self.paramdict['cb3'][1][1],:,:,:] ; h3, ss = self.relnormalize(h3, relprop_config)
		if DEBUG_INTERNAL: print("%-8s| %24s | %24s |  %24s | %8s"%(str("[3.1.h3]"),str(torch.max(h3).item()),str(torch.min(h3).item()),str(list(h3.shape)),str(h3.numel())))
		R = self.deconv1.relprop(R) ; R, ss = self.relnormalize(R, relprop_config)
		if DEBUG_INTERNAL:
			print("%-8s| %24s | %24s |  %24s | %8s"%(str("[3.2]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
	
		R = self.cblocks4.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[4]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		R = self.pool3.relprop(R)
		if DEBUG_INTERNAL: print("%-8s| %24s | %24s |  %24s | %8s"%(str("[4.1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
	
		factor = relprop_config['UNet3D']['concat_factors'][0]
		R = factor*h3 + (1-factor)*centre_crop_tensor(R, h3.shape).to(this_device)
		R = self.cblocks3.relprop(R,verbose=0); R, ss = self.relnormalize(R, relprop_config)
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[5]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		R = self.pool2.relprop(R)
		if DEBUG_INTERNAL:
			print("%-8s| %24s | %24s |  %24s | %8s"%(str("[5.1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
	
		factor2 = relprop_config['UNet3D']['concat_factors'][1]
		R = factor2*h2 + (1-factor2)* centre_crop_tensor(R, h2.shape).to(this_device)
		R = self.cblocks2.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		print("%-8s| %24s | %24s |  %24s | %8s"%(str("[6]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		R = self.pool1.relprop(R)
		if DEBUG_INTERNAL:
			print("%-8s| %24s | %24s |  %24s | %8s"%(str("[6.1]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
	
		factor3 = relprop_config['UNet3D']['concat_factors'][2]
		R = factor3*h1 + (1-factor3)*centre_crop_tensor(R, h1.shape).to(this_device)
		R = self.cblocks1.relprop(R); R, ss = self.relnormalize(R, relprop_config)
		if DEBUG_INTERNAL:
			print("%-8s| %24s | %24s |  %24s | %8s"%(str("[7]"),str(torch.max(R).item()),str(torch.min(R).item()),str(list(R.shape)),str(R.numel())))
		
		return R

class UNet3Db(UNet3D):
	# No of params 19 170 812
	# compare no of param 19 077 158
	def __init__(self, no_of_input_channel=6, with_LRP=False):
		super(UNet3Db, self).__init__(no_of_input_channel=no_of_input_channel, with_LRP=with_LRP, setup_lens_module=True)
		
		lens = LensModule(n_channel=no_of_input_channel)
		self.add_module("lens", lens)

		for x in self.modules(): x = torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count()))
		self._init_weight()

	def forward(self,x, save_for_relprop=True):
		if save_for_relprop: self.X = x
		h = self.lens(x, save_for_relprop=save_for_relprop)
		x = super(UNet3Db, self).forward(x*h, save_for_relprop=save_for_relprop)	
		return x 

	def forward_debug(self,x, save_for_relprop=True):
		print("UNet3Db.forward_debug()")
		if save_for_relprop: self.X = x
		h = self.lens.forward_debug(x, save_for_relprop=save_for_relprop)
		x = super(UNet3Db, self).forward_debug(x*h, save_for_relprop=save_for_relprop)	
		return x 

	def relprop(self,R):
		R = super(UNet3Db, self).relprop(R); ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5; R = R/ss
		R = self.lens.relprop(R/(1e-6+self.X)); ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5; R = R/ss
		return R

	def relprop_skip(self,R):
		R = super(UNet3Db, self).relprop_skip(R); ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5; R = R/ss
		R = self.lens.relprop(R/(1e-6+self.X)); ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5; R = R/ss
		return R

	def relprop_debug(self, R):
		R = super(UNet3Db, self).relprop_debug(R); ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5; R = R/ss
		R = self.lens.relprop_debug(R/(1e-6+self.X)); ss = sum((R.view(-1).detach().cpu().numpy())**2)**0.5; R = R/ss
		return R

def count_parameters(model, print_param=False):
	if print_param:
		for param in model.parameters(): print(param)
	num_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
	num_grad = sum(p.numel() for p in model.parameters())
	print("  networks.py. count_parameters()\n    with grad: %s, with or without: %s"%(num_with_grad, num_grad))
	return num_with_grad, num_grad
