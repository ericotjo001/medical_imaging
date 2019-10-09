from utils.utils import *
from models.SmallCNN import fast_setting

# no_of_classes=10

convblock_setting_conv_only = {
	'conv':{
		'in_channels': 3, 
		'out_channels':3,
		'kernel_size': 3,
		'stride':1,
		'padding':1,
		'dilation':1,
		'groups':1,
		'bias':False,
		'padding_mode':'zeros',
		'conv_name': str('conv_only')
	},
	'bn':{'use_batchNorm':False},
	'act':{'use_act': False}
}

convblock_setting_cb1_1 = fast_setting(3,36,3,'cb1_1')
convblock_setting_cb1_2 = fast_setting(36,3,5,'cb1_2',padding=2)
multiple_settings1 = [convblock_setting_cb1_1, convblock_setting_cb1_2]

convblock_setting_cb2_1 = fast_setting(3,36,5,'cb2_1',padding=2)
convblock_setting_cb2_2 = fast_setting(36,3,7,'cb2_2',padding=3)
multiple_settings2 = [convblock_setting_cb2_1, convblock_setting_cb2_2]

convblock_setting_cb3_1 = fast_setting(3,36,5,'cb3_1',padding=2)
convblock_setting_cb3_2 = fast_setting(36,3,7,'cb3_2',padding=3)
multiple_settings3 = [convblock_setting_cb3_1, convblock_setting_cb3_2]

conv_final_out = 3
convblock_setting_conv_final = {
	'conv':{
		'in_channels': 3, 
		'out_channels':conv_final_out,
		'kernel_size': 3,
		'stride':1,
		'padding':1,
		'dilation':1,
		'groups':1,
		'bias':False,
		'padding_mode':'zeros',
		'conv_name': 'conv_final'
	},
	'bn':{
		'use_batchNorm':False
		# 'use_batchNorm':True,
		# # 'num_features': 6, # same as convblock_setting['conv']['out_channels']
		# 'eps':1e-05,
		# 'momentum': 0.1,
		# 'affine':True,
		# 'track_running_stats':True,
		# 'bn_name':'bn_' + str(final_name)
	},
	'act':{
		'use_act': True,
		'activation_type':'LeakyReLU',
		'activation_name':'act_f' ,
		'negative_slope':0.01,
		'inplace':False
	}	
}