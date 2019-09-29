from utils.utils import *

conv_only_codename = 'conv_only'
cb1_codename = 'conv_000001'
cb2_codename = 'conv_000002'
cb3_codename = 'conv_000003'
cb4_codename = 'conv_000004'
cb5_codename = 'conv_000005'
cb6_codename = 'conv_000006'
final_name = 'conv_final' 
no_of_classes = 10

convblock_setting_conv_only = {
	'conv':{
		'in_channels': 3, 
		'out_channels':36,
		'kernel_size': 3,
		'stride':1,
		'padding':1,
		'dilation':1,
		'groups':1,
		'bias':True,
		'padding_mode':'zeros',
		'conv_name': str(conv_only_codename)
	},
	'bn':{'use_batchNorm':False},
	'act':{'use_act': False}
}

def fast_setting(inc,ouc,k, label_name, padding=1,stride=1, use_bn=True):
	setting = {
		'conv':{
			'in_channels': inc, 
			'out_channels':ouc,
			'kernel_size': k,
			'stride':stride,
			'padding':padding,
			'dilation':1,
			'groups':1,
			'bias':False,
			'padding_mode':'zeros',
			'conv_name': str(label_name)
		},
		'bn':{
			'use_batchNorm':use_bn,
			# 'num_features': 6, # same as convblock_setting['conv']['out_channels']
			'eps':1e-05,
			'momentum': 0.1,
			'affine':True,
			'track_running_stats':True,
			'bn_name':'bn' + str(label_name)
		},
		'act':{
			'use_act': True,
			'activation_type':'LeakyReLU',
			'activation_name':'act' + str(label_name),
			'negative_slope':0.01,
			'inplace':False
		}	
	}
	return setting

convblock_setting_cb1 = fast_setting(36,48,3,cb1_codename)
convblock_setting_cb2 = fast_setting(48,96,3,cb2_codename)
convblock_setting_cb3 = fast_setting(96,120,5,cb3_codename,padding=0,stride=2)
multiple_settings1 = [convblock_setting_cb1, convblock_setting_cb2, convblock_setting_cb3]

convblock_setting_cb4 = fast_setting(120,124,3,cb4_codename)
convblock_setting_cb5 = fast_setting(124,96,3,cb5_codename)
convblock_setting_cb6 = fast_setting(96,64,3,cb6_codename,padding=0,stride=2)
multiple_settings2 = [convblock_setting_cb4, convblock_setting_cb5, convblock_setting_cb6]

convblock_setting_cb7 = fast_setting(64,48,3,cb4_codename)
convblock_setting_cb8 = fast_setting(48,48,5,cb5_codename)
convblock_setting_cb9 = fast_setting(48,24,3,cb6_codename,padding=1,stride=1)
multiple_settings3 = [convblock_setting_cb7, convblock_setting_cb8, convblock_setting_cb9]

convblock_setting_conv_final = {
	'conv':{
		'in_channels': 24, 
		'out_channels':no_of_classes,
		'kernel_size': 4,
		'stride':1,
		'padding':0,
		'dilation':1,
		'groups':1,
		'bias':False,
		'padding_mode':'zeros',
		'conv_name': final_name
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