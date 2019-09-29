from utils.utils import *
from dataio.dataISLES2017 import ISLES2017mass
import models.networks as net
import utils.vis as vi

from pipeline.evaluation import get_modalities_0001
from pipeline.evaluation import generic_data_loading

# DEBUG_lrp_relprop_one = 1

def lrp_UNet3D_overfit(config_data):
	print("pipeline/lrp.py. lrp_UNet3D_overfit().")

	with torch.no_grad():
		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		model_type = 'UNet3D'
		modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
		main_model_fullpath = os.path.join(model_dir,config_data['model_label_name'] + '.model') 
		
		if model_type == 'UNet3D': this_net = net.UNet3D(no_of_input_channel=no_of_input_channels, with_LRP=True)
		if os.path.exists(main_model_fullpath): this_net = this_net.load_state(config_data); print("Load existing model...")
		this_net.eval()# .cpu()

		for_evaluation = generic_data_loading(config_data)

		print("\n========== starting LRP evaluation: ==========")
		for case_number in for_evaluation:
			print("case number:%s"%(str(case_number)))
			x, labels = for_evaluation[case_number] # shapes are like torch.Size([1, 6, 19, 192, 192]) torch.Size([256, 256, 24])
			# x = x.cpu()
			# labels = labels.cpu()
			
			s, sx = labels.shape, np.array(x.shape)
			no_of_modalities = sx[1]
			print("  s=%s,sx=%s, no_of_modalities=%s"%(str(np.array(s)),str(sx),str(no_of_modalities)))

			R = this_net(x).contiguous()
			outputs = torch.argmax(this_net(x).contiguous(),dim=1)
			outputs = outputs.squeeze().permute(2,1,0).to(torch.float)
			outputs_label = interp3d(outputs,tuple(labels.shape), mode='nearest')
			y = outputs_label.detach().cpu().numpy()
			
			if DEBUG_lrp_relprop_one: R = this_net.relprop_debug(R); break
			R = this_net.relprop(R).squeeze()
			Rc = np.zeros(shape=(no_of_modalities,)+s)
			for c in range(x.shape[1]):
				Rc_part = centre_crop_tensor(R[c], sx[2:] ).permute(2,1,0)
				Rc_part = interp3d(Rc_part,s).detach().cpu().numpy()
				# print("  np.max(Rc_part) = %s, np.min(Rc_part)=%s"%(str(np.max(Rc_part)),str(np.min(Rc_part))))
				Rc[c] = Rc_part

			Rmax, Rmin = np.max(Rc), np.min(Rc)
			print("  Rc.shape = %s [%s]\n  y.shape = %s [%s]"%(str(Rc.shape),str(type(Rc)),str(y.shape),str(type(y))))
			print("  Rmax = %s, Rmin = %s"%(str(Rmax),str(Rmin)))

			save_one_lrp_output(case_number,y,Rc, config_data)

def lrp_UNet3D_overfit_visualizer(config_data):
	case_type = 'training'
	case_number = config_data['misc']['case_number']
	data_modalities = config_data['data_modalities']
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	lrp_dir = os.path.join(model_dir,'output_lrp')
	lrp_fullpath = os.path.join(lrp_dir,'yR_'+case_type+'_'+str(case_number)+'.lrp')

	print("data_modalities:%s"%(str(data_modalities)))
	pkl_file = open(lrp_fullpath, 'rb')
	y, Rc = pickle.load(pkl_file)
	pkl_file.close() 
	print("  Rc.shape = %s [%s]\n  y.shape = %s [%s]"%(str(Rc.shape),str(type(Rc)),str(y.shape),str(type(y))))
	
	ISLESDATA = ISLES2017mass()
	ISLESDATA.directory_path = config_data['data_directory']['dir_ISLES2017']
	one_case = ISLESDATA.load_one_case(case_type, str(case_number), data_modalities)

	vis = vi.SlidingVisualizer()
	vis.do_show = True
	# vis.vis2_lrp_training(one_case,y,Rc,data_modalities)
	args = (one_case,y,Rc,data_modalities)
	p = Pool(2)
	p.map(vis.vis_lrp_training, [args+(1,),args+(2,)] )

def save_one_lrp_output(case_number,y,Rc, config_data,case_type='training'):
	'''
	y is output predicted by the network.
	Rc is LRP output
	'''
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	lrp_dir = os.path.join(model_dir,'output_lrp')
	if not os.path.exists(lrp_dir): os.mkdir(lrp_dir)
	lrp_fullpath = os.path.join(lrp_dir,'yR_'+case_type+'_'+str(case_number)+'.lrp')

	output = open(lrp_fullpath, 'wb')
	pickle.dump([y,Rc], output)
	output.close()