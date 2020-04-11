CONFIG_FILE = {
	'working_dir':"D:/Desktop@D/meim2venv/meim3",
	'relative_checkpoint_dir':'checkpoints',
	'data_directory':{
		'dir_ISLES2017':"D:/Desktop@D/meim2venv/meim3/data/isles2017",
	},
	'data_submode':'load_many_cases_type0003',
	'data_modalities':['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT'],
	'model_label_name': 'UNet3D_AYXXX2',
	'training_mode': 'UNet3D',
	'evaluation_mode': 'UNet3D_overfit',
	'visual_mode':'lrp_UNet3D_overfit_visualizer',
	'lrp_mode': 'lrp_UNet3D_overfit',
	'debug_test_mode':"test_load_many_ISLES2017",
	
	'basic':{
		'batch_size' : 1,
		'n_epoch' : 2,
		'save_model_every_N_epoch': 1,
		'keep_at_most_n_latest_models':4
	},
	'dataloader':{
		'resize' : [48,48,19] # [192,192,19]
	},
	'learning':{
		'mechanism':"adam",
		'momentum':0.9,
		'learning_rate':0.0002,
		'weight_decay':0.00001,
		'betas':[0.5,0.9],
		},
	'normalization': {
		'ADC_source_min_max' : [None, None], # "[0, 5000]",
		'MTT_source_min_max' : [None, None],
		'rCBF_source_min_max' : [None, None],
		'rCBV_source_min_max' : [None, None],
		'Tmax_source_min_max' : [None, None],
		'TTP_source_min_max' : [None, None],
		
		'ADC_target_min_max' : [0,1],
		'MTT_target_min_max' : [0,1],
		'rCBF_target_min_max' : [0,1],
		'rCBV_target_min_max' : [0,1],
		'Tmax_target_min_max' : [0,1],
		'TTP_target_min_max' : [0,1],
	},
	'augmentation': {
		'type': 'no_augmentation',
		'number_of_data_augmented_per_case': 10,
	},
	'LRP':{
		'relprop_config':{
			'mode': 'UNet3D_standard',
			'normalization': 'raw',
			'UNet3D': {
				'concat_factors': [0.5,0.5,0.5]
			},
			'fraction_pass_filter' :{
				"positive":[0.0,0.6],
				"negative":[-0.6,-0.0]		
			},
			'fraction_clamp_filter' :{
				"positive":[0.0,0.6],
				"negative":[-0.6,-0.0]		
			},
		},
		'filter_sweeper' :{
			'submode' : '0001',
			'case_numbers': [4,27]
		}
	},
	'misc': {
		'case_number':1
	}
}

'''
Some available default values to plug into CONFIG_FILE

	'learning':{
		'mechanism':"SGD",
		'momentum':"0.9",
		'learning_rate':"0.01",
		'weight_decay':"0.00001"
		},
'''