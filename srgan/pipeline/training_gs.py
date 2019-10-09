from utils.utils import *


def adjust_attr(model, name, add_value, multiplier):
	attrlist = name.split('.')
	# print(attrlist)
	temp = model
	for attr in attrlist:
		temp = getattr(temp, attr)
	temp.data = multiplier*(temp.data + add_value)
	# print(temp)
	return model



def gram_schmidt_process(gs_net, this_net, config_data, verbose=0):
	epsil = 1e-8
	cs = torch.nn.CosineSimilarity(dim=0, eps=epsil)

	param_store1={} # for checking
	if verbose>=100:
		print("gram_schmidt_process()")
		count,count2 = 0,0

	for temp1, temp2 in zip(gs_net.named_parameters(), this_net.named_parameters()):
		name, param = temp1
		name2, param2 = temp2
		param_store1[name] = param.clone()
		bmax = torch.max(cs(param2,param2),epsil*torch.ones(param2.shape).to(device=this_device))
		gs_net = adjust_attr(gs_net, name, - param2 * (cs(param,param2)/bmax),1.)
		if verbose>=100:
			count+=1
			if len(param.shape)>1:
				print(" [%s] shape : %-40s"%(str(count),str(param.shape)))
			else:
				csvalue = cs(param,param2).item()
				print(" [%s] shape*: %-40s | %10s"%(str(count),str(param.shape),str(csvalue)))
				if csvalue == 0.: print("    ",name)


	for name, param in gs_net.named_parameters():
		check = np.any(param.detach().cpu().numpy()==param_store1[name].detach().cpu().numpy())
		# assert(check)
		if verbose>=100:
			count2+=1
			print(" [%s] %s "%(str(count2),str(check)))

	from pipeline.training_aux import adjust_config_data_for_helper_network

	path_to_gs_folder = os.path.join(config_data['working_dir'],config_data['relative_checkpoint_dir'],\
		config_data['model_label_name'])

	if not os.path.exists(path_to_gs_folder): os.mkdir(path_to_gs_folder)
	gs_net.elapsed = 0.
	gs_net = gs_net.perform_routine_end(gs_net, config_data)
	