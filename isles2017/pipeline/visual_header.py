from utils.utils import *
import utils.evalobj as ev
import utils.vis as vi

def load_lrp_sweep(lrpd_file_fullpath):
	"""
	output_dictionary looks like this:
	it is shown in the following format: key| array shape
	                                                       raw | (6, 192, 192, 19)
	                                                        OT | (192, 192, 19)
	                                                         y | (192, 192, 19)
	         ('fraction_pass_filter', (-0.3, 0.0), (0.0, 0.3)) | (6, 192, 192, 19)
	         ('fraction_pass_filter', (-0.7, 0.0), (0.0, 0.7)) | (6, 192, 192, 19)
	        ('fraction_pass_filter', (-0.5, -0.2), (0.2, 0.5)) | (6, 192, 192, 19)
	        ('fraction_pass_filter', (-0.9, -0.2), (0.2, 0.9)) | (6, 192, 192, 19)
	        ('fraction_pass_filter', (-0.7, -0.4), (0.4, 0.7)) | (6, 192, 192, 19)
	        ('fraction_pass_filter', (-0.9, -0.6), (0.6, 0.9)) | (6, 192, 192, 19)
	        ('fraction_clamp_filter', (-0.3, 0.0), (0.0, 0.3)) | (6, 192, 192, 19)
	        ('fraction_clamp_filter', (-0.7, 0.0), (0.0, 0.7)) | (6, 192, 192, 19)
	       ('fraction_clamp_filter', (-0.5, -0.2), (0.2, 0.5)) | (6, 192, 192, 19)
	       ('fraction_clamp_filter', (-0.9, -0.2), (0.2, 0.9)) | (6, 192, 192, 19)
	       ('fraction_clamp_filter', (-0.7, -0.4), (0.4, 0.7)) | (6, 192, 192, 19)
	       ('fraction_clamp_filter', (-0.9, -0.6), (0.6, 0.9)) | (6, 192, 192, 19)

	Usage examples:
	  pipeline/visual.py. lrp_UNet3D_filter_sweep_visualizer()
	"""
	pkl_file = open(lrpd_file_fullpath, 'rb')
	lrp_output_dictionary = pickle.load(pkl_file)
	pkl_file.close() 
	return lrp_output_dictionary

def lrp_UNet3D_filter_sweep_0003_visualizer_aux(config_data,model_label_names_collection, filter_name, color_list, verbose=0):
	dict_of_nsXY_bLRPtoOT = {}
	dict_of_nsXY_bLRPtobx = {}
	dict_of_XY_bLRPtoOT = {}
	dict_of_XY_bLRPtobx = {}
	for collection_name in model_label_names_collection:
		model_label_names = model_label_names_collection[collection_name]
		XY_bLRPtoOT = {}
		XY_bLRPtobx = {}
		for model_label_name in model_label_names:
			case_number = config_data['misc']['case_number']
			print("model_label_name:%s case number:%s"%(model_label_name,str(case_number)))
			case_type = 'training'
			data_modalities = config_data['data_modalities']
			model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],model_label_name)
			output_folder_name = 'lrp_filter_sweep_mode_0003'
			lrp_dir = os.path.join(model_dir,output_folder_name)

			config_data['model_label_name'] = model_label_name

			evLRP = ev.EvalLRP(config_data, output_folder_name, save_name='EvalLrp_InterpQuant.evlrpiq')
			evLRP = evLRP.load_object(evLRP.save_fullpath)
			filter_data = select_data_by_filter(filter_name, evLRP, verbose=verbose)

			# preparing data into DictionaryWithNumericalYArray() compatible form
			for filter_interval in filter_data:
				if type(filter_interval)==type(''): thekey = filter_interval
				else: thekey = filter_interval[1]
				if not (thekey in XY_bLRPtoOT): XY_bLRPtoOT[thekey] = []
				if not (thekey in XY_bLRPtobx): XY_bLRPtobx[thekey] = []
				for this_unit in filter_data[filter_interval]:		
					if case_number == this_unit['case_number']: 
						for x,y in zip(this_unit["bLRPtoOT"],this_unit["bLRPtobx"]):
							if type(filter_interval)==type(''): thekey = filter_interval
							else: thekey = filter_interval[1]
							XY_bLRPtoOT[thekey].append(x)
							XY_bLRPtobx[thekey].append(y)
							
		dict_of_XY_bLRPtoOT, dm_bLRPtoOT = arrange_dictionary_for_plotting(XY_bLRPtoOT,dict_of_nsXY_bLRPtoOT, dict_of_XY_bLRPtoOT, collection_name)
		dict_of_XY_bLRPtobx, dm_bLRPtobx = arrange_dictionary_for_plotting(XY_bLRPtobx,dict_of_nsXY_bLRPtobx, dict_of_XY_bLRPtobx, collection_name)

	print_dictionary_0003_aux0001(dict_of_XY_bLRPtoOT,'dict_of_XY_bLRPtoOT')
	print_dictionary_0003_aux0001(dict_of_XY_bLRPtobx,'dict_of_XY_bLRPtobx')

	# title_text = filter_name + '\nmetric:'+str('bLRPtoOT') + '\ncase number:'+str(case_number)
	# dm_bLRPtoOT.scatter_layered_normal_scatter_list(dict_of_nsXY_bLRPtoOT, title=title_text ,xlim=None, ylim=None,marker='x',size=10)
	# dm_bLRPtoOT.layered_boxplots(dict_of_XY_bLRPtoOT, color_list, shift_increment=0.75, x_index_increment=4, title=title_text)

	# title_text = filter_name + '\nmetric:'+str('bLRPtobx') + '\ncase number:'+str(case_number)
	# dm_bLRPtobx.scatter_layered_normal_scatter_list(dict_of_nsXY_bLRPtobx, title=title_text ,xlim=None, ylim=None,marker='x',size=10)
	# dm_bLRPtobx.layered_boxplots(dict_of_XY_bLRPtobx, color_list, shift_increment=0.75, x_index_increment=4, title=title_text)

def print_dictionary_0003_aux0001(dict_of_XY, title_name):
	print(title_name)
	for this_size in dict_of_XY:
		print("  %s"%(str(this_size)))
		for filter_interval in dict_of_XY[this_size]:
			mean_value = np.mean(dict_of_XY[this_size][filter_interval])
			print("    %-12s|%s"%(str(filter_interval),str(round(mean_value,4))))

def arrange_dictionary_for_plotting(XY, dict_of_nsXY, dict_of_XY, collection_name):
	dm = vi.DictionaryWithNumericalYArray()
	dm.this_dictionary = XY
	dict_of_XY[collection_name] = XY
	dm.normal_scatter_mapping_index()
	nsXY = dm.get_normal_scatter_list(mu=0,sigma=0.2, verbose=0)
	dict_of_nsXY[collection_name] = nsXY
	return dict_of_XY, dm

def select_data_by_filter(filter_name, evLRP, verbose=0):
	if filter_name == 'fraction_pass_filter':
		if verbose >=100: evLRP.print_pfilter_data_0003()
		filter_data = evLRP.pfilter_data_0003
	elif filter_name == 'fraction_clamp_filter':
		if verbose >=100: evLRP.print_cfilter_data_0003()
		filter_data = evLRP.cfilter_data_0003
	return filter_data


def lrp_UNet3D_filter_sweep_0002_visualizer_aux(config_data,model_label_names_collection, filter_name, color_list, verbose=0):
	dict_of_nsXY = {}
	dict_of_XY = {}
	for collection_name in model_label_names_collection:
		model_label_names = model_label_names_collection[collection_name]
		XY = {}
		for model_label_name in model_label_names:
			print("model_label_name:%s"%(model_label_name))
			case_type = 'training'
			data_modalities = config_data['data_modalities']
			model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],model_label_name)
			output_folder_name = 'lrp_filter_sweep_mode_0002'
			lrp_dir = os.path.join(model_dir,output_folder_name)

			config_data['model_label_name'] = model_label_name

			evLRP = ev.EvalLRP(config_data, output_folder_name, save_name='EvalLrp.evlrp')
			evLRP = evLRP.load_object(evLRP.save_fullpath)
			if filter_name == 'fraction_pass_filter':
				if verbose >=100: evLRP.print_pfilter_data_0002()
				filter_data = evLRP.pfilter_data_0002
			elif filter_name == 'fraction_clamp_filter':
				if verbose >=100:  evLRP.print_cfilter_data_0002()
				filter_data = evLRP.cfilter_data_0002

			# preparing data into DictionaryWithNumericalYArray() compatible form
			for filter_interval in filter_data:
				if not (filter_interval[1] in XY): XY[filter_interval[1]] =[]
				for this_unit in filter_data[filter_interval]:
					xsubset = this_unit['normalized_mean']['values']
					for x in xsubset:
						XY[filter_interval[1]].append(x)

		dm = vi.DictionaryWithNumericalYArray()
		dm.this_dictionary = XY
		dict_of_XY[collection_name] = XY
		dm.normal_scatter_mapping_index()
		nsXY = dm.get_normal_scatter_list(mu=0,sigma=0.2, verbose=0)
		dict_of_nsXY[collection_name] = nsXY
	dm.scatter_layered_normal_scatter_list(dict_of_nsXY, title=filter_name ,xlim=None, ylim=None,marker='x',size=3)
	dm.layered_boxplots(dict_of_XY, color_list, shift_increment=0.75, x_index_increment=4, title=filter_name)


