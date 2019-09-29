DEBUG_VERSION = 1


if DEBUG_VERSION:
	# pipeline/training,py
	DEBUG_TRAINING_CASE_NUMBERS = range(1,10) # default is None
	DEBUG_TRAINING_LOOP = 0
	DEBUG_TRAINING_DATA_LOADING = 0
	# DEBUG_TRAINING_LABELS_RESHAPE = 0 # if memory usage too large without reshaping during tests. Set to "resize": "[124,124,9]" when testing


	# pipeline/test.py
	DEBUG_EVAL_TRAINING_CASE_NUMBERS = [1,6] # range(1,10) # default is None # for training
	DEBUG_EVAL_TEST_CASE_NUMBERS = [1,2,3] # None # range(1,10) # default is None # for test
	DEBUG_EVAL_LOOP = 0


	# pipeline/lrp.py
	DEBUG_lrp_relprop_one = 0


	# dataio/dataISLES2017.py
	DEBUG_dataISLES2017_load_type0003_aug = 0
	DEBUG_dataISLES2017_AUG_NUMBER = 0 # default 0


	# models/networks.py
	UNet3D_DEBUG = 1 


	# models/networks_components.py
	DEBUG_networks_components_COMPONENT = 0

else:
	# pipeline/training,py
	DEBUG_TRAINING_CASE_NUMBERS = None 
	DEBUG_TRAINING_LOOP = 0
	DEBUG_TRAINING_DATA_LOADING = 0
	# DEBUG_TRAINING_LABELS_RESHAPE = 0 # if memory usage too large without reshaping during tests. Set to "resize": "[124,124,9]" when testing


	# pipeline/test.py
	DEBUG_EVAL_TRAINING_CASE_NUMBERS = None # for training
	DEBUG_EVAL_TEST_CASE_NUMBERS = None # range(1,10) # default is None # for test
	DEBUG_EVAL_LOOP = 0


	# pipeline/lrp.py
	DEBUG_lrp_relprop_one = 0


	# dataio/dataISLES2017.py
	DEBUG_dataISLES2017_load_type0003_aug = 0
	DEBUG_dataISLES2017_AUG_NUMBER = 0 # default 0


	# models/networks.py
	UNet3D_DEBUG = 0


	# models/networks_components.py
	DEBUG_networks_components_COMPONENT = 0