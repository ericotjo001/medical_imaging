from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
import utils.loss as uloss
import utils.evalobj as ev
import models.networks as net

DEBUG_CASE_NUMBERS = None #  range(1,5) # default is None
DEBUG_TRAINING_LOOP = 0
DEBUG_LABELS_RESHAPE = 0 # if memory usage too large without reshaping during tests. Set to "resize": "[124,124,9]" when testing
DEBUG_DATA_LOADING = 0

n_NSC, n_NSCy = 2, 1 # number of nonspatial dimension

def training_segnet(config_data):
	print("training_segnet()")
	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0003(case_type, case_numbers, config_data,normalize=True)
	if DEBUG_DATA_LOADING: return
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=config_data['basic_1']['batch_size'], 
		shuffle=True)
	print("  trainloader loaded")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.segnet()

	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	this_net.training_cycle = this_net.training_cycle + 1
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	criterion = nn.CrossEntropyLoss()
	if config_data['learning']['mechanism'] == 'SGD':
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], 
			momentum=config_data['learning']['momentum'], weight_decay=config_data['learning']['weight_decay'])
	cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=10)

	for i_epoch in range(config_data['basic_1']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			# optimizer.zero_grad()
			x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
			labels = data[1].to(this_device).to(torch.float)

			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
		
			if DEBUG_TRAINING_LOOP: outputs = this_net.forward_debug(x);break

			outputs = this_net(x).contiguous() # .view((config_data['basic_1']['batch_size'],number_of_classes,)+resize_shape)
			
			loss = criterion(outputs, labels.to(torch.int64))
			cetracker.store_loss(loss) # tracking progress
			loss.backward()
			optimizer.step()
		this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data)
		this_net.save_models(this_net, config_data)
	cetracker.save_loss_plot()

def training_PSPNet(config_data):
	print("training_PSPNet()")
	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0003(case_type, case_numbers, config_data,normalize=True)
	if DEBUG_DATA_LOADING: return
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=config_data['basic_1']['batch_size'], 
		shuffle=True)
	print("  trainloader loaded")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.PSPNet()
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	this_net.training_cycle = this_net.training_cycle + 1
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	criterion, aux_crit = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
	if config_data['learning']['mechanism'] == 'SGD':
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], 
			momentum=config_data['learning']['momentum'], weight_decay=config_data['learning']['weight_decay'])
	cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=10)

	for i_epoch in range(config_data['basic_1']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			# optimizer.zero_grad()
			x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
			labels = data[1].to(this_device).to(torch.float)

			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
		
			if DEBUG_TRAINING_LOOP: outputs = this_net.forward_debug(x);break

			outputs, out_aux = this_net(x) # .view((config_data['basic_1']['batch_size'],number_of_classes,)+resize_shape)
			
			loss = criterion(outputs, labels.to(torch.int64))
			cetracker.store_loss(loss) # tracking progress
			loss.backward(retain_graph=True)
			loss2 = aux_crit(out_aux, labels.to(torch.int64))
			loss2.backward()
			optimizer.step()
		this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data)
		this_net.save_models(this_net, config_data)
	cetracker.save_loss_plot()

def training_UNet3D(config_data):
	print("training_UNet3D()")
	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0003(case_type, case_numbers, config_data,normalize=True)
	if DEBUG_DATA_LOADING: return
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=config_data['basic_1']['batch_size'], 
		shuffle=True)
	print("  trainloader loaded")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.UNet3D()

	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	this_net.training_cycle = this_net.training_cycle + 1
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	criterion = nn.CrossEntropyLoss()
	if config_data['learning']['mechanism'] == 'SGD':
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], 
			momentum=config_data['learning']['momentum'], weight_decay=config_data['learning']['weight_decay'])	
	cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=10)

	for i_epoch in range(config_data['basic_1']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
			labels = data[1].to(this_device).to(torch.float)

			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
		
			if DEBUG_TRAINING_LOOP: outputs = this_net.forward_debug(x);break

			outputs = this_net(x).contiguous() # .view((config_data['basic_1']['batch_size'],number_of_classes,)+resize_shape)
			# outputs = batch_interp3d(outputs,tuple(labels.shape)[1:], mode='nearest').to(device=this_device).to(torch.float)

			loss = criterion(outputs, labels.to(torch.int64))
			cetracker.store_loss(loss) # tracking progress
			loss.backward()
			optimizer.step()
		this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data)
		this_net.save_models(this_net, config_data)
		cetracker.save_loss_plot()

def training_FCN8like(config_data):
	print("training_FCN8like().")
	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'

	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0003(case_type, case_numbers, config_data,normalize=True)
	if DEBUG_DATA_LOADING: return
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=config_data['basic_1']['batch_size'], 
		shuffle=True)
	print("  trainloader loaded")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.FCN8like()
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	this_net.training_cycle = this_net.training_cycle + 1
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	criterion = nn.CrossEntropyLoss()
	if config_data['learning']['mechanism'] == 'SGD':
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], 
			momentum=config_data['learning']['momentum'], weight_decay=config_data['learning']['weight_decay'])
	cetracker = ev.CrossEntropyLossTracker(config_data, display_every_n_minibatchs=10)


	for i_epoch in range(config_data['basic_1']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
			labels = data[1].to(this_device).to(torch.float)

			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
		
			if DEBUG_TRAINING_LOOP: outputs = this_net.forward_debug(x);break

			outputs = this_net(x).contiguous() # .view((config_data['basic_1']['batch_size'],number_of_classes,)+resize_shape)
			# outputs = batch_interp3d(outputs,tuple(labels.shape)[1:], mode='nearest').to(device=this_device).to(torch.float)

			loss = criterion(outputs, labels.to(torch.int64))
			cetracker.store_loss(loss) # tracking progress
			loss.backward()
			optimizer.step()
		this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data)
		this_net.save_models(this_net, config_data)
	cetracker.save_loss_plot()


########################################################################################
# Here onwards for testing only
########################################################################################

def training_FCN_1(config_data):
	print("training_FCN_1().")

	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0002(case_type, case_numbers, config_data,normalize=True)
	# trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, batch_size=config_data['basic_1']['batch_size'], shuffle=True)
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, batch_size=1, shuffle=True)
	'''
	with batch size fixed to 1, there should not be any batch norm.
	'''
	print("  trainloader loaded.")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.FCN(device=this_device)
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	this_net.training_cycle = this_net.training_cycle + 1
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	# criterion = uloss.SoftDiceLoss()
	criterion = nn.CrossEntropyLoss()
	if config_data['learning']['mechanism'] == 'SGD':
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], momentum=config_data['learning']['momentum'])

	resize_shape = tuple(config_data['FCN_1']['resize'])
	for i_epoch in range(config_data['basic_1']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			x = data[0].to(this_device).to(torch.float) # interp3d() resizes only x. labels remains in the original shape.
			labels = data[1].to(this_device).to(torch.float)
			if DEBUG_LABELS_RESHAPE: labels = batch_no_channel_interp3d(labels,resize_shape,mode='nearest').to(device=this_device)
			
			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
			
			if DEBUG_TRAINING_LOOP: 
				outputs = this_net.forward_debug(x) # .view((config_data['basic_1']['batch_size'],number_of_classes,)+resize_shape)
				check_network_info_training_FCN_1(this_net, x, outputs, labels); break
				outputs = batch_interp3d(outputs,tuple(labels.shape)[1:], mode='nearest').to(device=this_device).to(torch.float)
			
			outputs = this_net(x).contiguous() # .view((config_data['basic_1']['batch_size'],number_of_classes,)+resize_shape)
			outputs = batch_interp3d(outputs,tuple(labels.shape)[1:], mode='nearest').to(device=this_device).to(torch.float)

			loss = criterion(outputs, labels.to(torch.int64))
			loss.backward()
			optimizer.step()
		this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data)
		this_net.save_models(this_net, config_data)

def training_basic_1(config_data):
	print("training_basic_1().")

	case_numbers = range(1,49)
	if DEBUG_CASE_NUMBERS is not None: case_numbers = DEBUG_CASE_NUMBERS
	case_type = 'training'
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['dir_ISLES2017']
	ISLESDATA.load_many_cases_type0001(case_type, case_numbers, config_data,normalize=True)
	trainloader = DataLoader(dataset=ISLESDATA, num_workers=0, 
		batch_size=config_data[config_data['training_mode']]['batch_size'], 
		shuffle=True)
	print("  trainloader loaded.")

	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
	this_net = net.CNN_basic(device=this_device)
	if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
	this_net.training_cycle = this_net.training_cycle + 1
	this_net.write_diary(config_data)
	net.count_parameters(this_net)

	# criterion = uloss.SoftDiceLoss()
	criterion = nn.CrossEntropyLoss()
	threshold = 0.9
	if config_data['learning']['mechanism'] == 'SGD':
		optimizer = optim.SGD(this_net.parameters(), lr=config_data['learning']['learning_rate'], momentum=config_data['learning']['momentum'])
	
	for i_epoch in range(config_data['basic_1']['n_epoch']):
		this_net.start_timer()
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			x = data[0].to(this_device).to(torch.float)
			labels = data[1].to(this_device).to(torch.float)
			x.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC) # N,C,W,H,D to N,C,D,H,W
			labels.transpose_(0+n_NSCy,1+n_NSCy).transpose_(1+n_NSCy,2+n_NSCy).transpose_(0+n_NSCy,1+n_NSCy)
			outputs = this_net(x)
			if DEBUG_TRAINING_LOOP: this_net.forward_debug(x); check_network_info_training_basic_1(this_net, x, labels, outputs); break
			
			loss = criterion(outputs, labels.to(torch.int64))
			loss.backward()
			optimizer.step()
			this_net.stop_timer()
		if DEBUG_TRAINING_LOOP: break
		this_net.latest_epoch = this_net.latest_epoch + 1
		this_net.write_diary_post_epoch(config_data)
		this_net.save_models(this_net, config_data)


def check_network_info_training_basic_1(model, x, labels, outputs):
	print("\n======== check_network_info_training_basic_1() ========")
	print('  [REAL]\n  x.shape=%s\n  labels.shape=%s\n  outputs.shape=%s'%(str(x.shape),str(labels.shape),str(outputs.shape)))
	
	x_fake = np.random.normal(0,1,size=(4,6,256,256,25))
	x_fake = torch.Tensor(x_fake).to(this_device)
	y = model.forward_debug(x_fake)
	x_fake.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC)
	print('  [TEST]\n  x_fake.shape=%s\n  y.shape=%s'%(str(x_fake.shape),str(y.shape)))
	
	x_fake2 = np.random.normal(0,1,size=(4,6,192,192,17))
	x_fake2 = torch.Tensor(x_fake2).to(this_device)
	x_fake2.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC)
	y2 = model.forward_debug(x_fake2)
	print('  [TEST2]\n  x_fake.shape=%s\n  y.shape=%s'%(str(x_fake2.shape),str(y2.shape)))
	print("\n=======================================================")
	return

def check_network_info_training_FCN_1(model, x, outputs, labels):
	print("\n========= check_network_info_training_FCN_1() =========")
	print('  [REAL]\n  x.shape=%s\n  labels.shape=%s\n  outputs.shape=%s'%(str(x.shape),str(labels.shape),str(outputs.shape)))

	x_fake = np.random.normal(0,1,size=(4,6,72,72,25))
	x_fake = torch.Tensor(x_fake).to(this_device)
	x_fake.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC)
	y = model.forward_debug(x_fake)
	print('  [TEST]\n  x_fake.shape=%s\n  y.shape=%s'%(str(x_fake.shape),str(y.shape)))
	
	x_fake2 = np.random.normal(0,1,size=(4,6,17,192,192))
	x_fake2 = torch.Tensor(x_fake2).to(this_device)
	x_fake.transpose_(0+n_NSC,1+n_NSC).transpose_(1+n_NSC,2+n_NSC).transpose_(0+n_NSC,1+n_NSC)
	y2 = model.forward_debug(x_fake2)
	print('  [TEST2]\n  x_fake.shape=%s\n  y.shape=%s'%(str(x_fake2.shape),str(y2.shape)))
	print("Notice that the output shape is highly sensitive to the network configuration. Edit accordingly.")
	print("\n=========================================================")
	return

