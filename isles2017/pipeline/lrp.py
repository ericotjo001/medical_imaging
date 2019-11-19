from pipeline.lrp_header import *
from pipeline.lrp_custom_options import get_filter_sweep_options


def lrp_UNet3D_overfit(config_data, verbose=0):
	print("pipeline/lrp.py. lrp_UNet3D_overfit().")
	normalization_mode = str(config_data['LRP']['relprop_config']['normalization'])
	output_lrp_folder_name = 'output_lrp_' +  normalization_mode

	lrp_UNet3D_overfit_options(config_data, output_lrp_folder_name=output_lrp_folder_name, verbose=0)

def lrp_UNet3D_filter_sweeper(config_data, verbose=0):
	print("pipeline/lrp.py. lrp_UNet3D_filter_sweeper().")
	if config_data['LRP']['filter_sweeper']['submode'] == '0001': lrp_UNet3D_filter_sweeper_0001(config_data,verbose=0)
	elif config_data['LRP']['filter_sweeper']['submode'] == '0002': lrp_UNet3D_filter_sweeper_0002(config_data,verbose=0)
	elif config_data['LRP']['filter_sweeper']['submode'] == '0003': lrp_UNet3D_filter_sweeper_0003(config_data,verbose=0)
	else: raise Exception('Invalid mode!')
############################ sub-modes ################################

def lrp_UNet3D_filter_sweeper_0003(config_data,verbose=0):
	print("  lrp_UNet3D_filter_sweeper_0003")
	"""
	  NOTE: 
	    this function is only to be run after lrp_UNet3D_filter_sweeper_0002 output has been generated
		as such, the folder it tries to find is 'lrp_filter_sweep_mode_0002'
	"""

	from pipeline.lrp import get_modalities_0001
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
	modalities_dict, no_of_input_channels = get_modalities_0001(config_data)	
	# {0: 'ADC', 1: 'MTT', 2: 'rCBF', 3: 'rCBV', 4: 'Tmax', 5: 'TTP'}
	# print(modalities_dict)

	import utils.evalobj as ev
	output_folder_name = 'lrp_filter_sweep_mode_0003'
	evLRP = ev.EvalLRP(config_data, output_folder_name, save_name='EvalLrp_InterpQuant.evlrpiq')
	
	case_numbers = config_data['LRP']['filter_sweeper']['case_numbers']
	print("    case_numbers:%s"%(str(case_numbers)))

	for case_number in case_numbers:
		x1, lrp_output_dictionary = lrp_UNet3D_filter_sweeper_0003_prepare_data(config_data,\
			case_number,modalities_dict, verbose=0)
		evLRP.process_one_case(case_number, lrp_output_dictionary, x1=x1, processing_mode='0003', verbose=0)		
		if DEBUG_EvalLRP_0003_LOOP: return
	
	print("========= save & review ========================")
	evLRP.save_object(evLRP.save_fullpath)
	print("Done.")

def lrp_UNet3D_filter_sweeper_0002(config_data,verbose=0):
	print("  submode: 0002")
	mode = '0002'
	diary_header_text = 'lrp_UNet3D_filter_sweeper_0002()'
	lrp_UNet3D_filter_sweeper_intermediate(config_data, mode, verbose=verbose)

def lrp_UNet3D_filter_sweeper_0001(config_data,verbose=0):
	print("  submode: 0001")
	mode = '0001'
	diary_header_text = 'lrp_UNet3D_filter_sweeper_0001()'
	lrp_UNet3D_filter_sweeper_intermediate(config_data, mode, verbose=verbose)

def lrp_UNet3D_filter_sweeper_intermediate(config_data, mode, diary_header_text=None, verbose=0):
	dice_loss = uloss.SoftDiceLoss()
	relprop_config = config_data['LRP']['relprop_config']
	case_numbers = config_data['LRP']['filter_sweeper']['case_numbers']
	case_type = 'training'
	canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']

	with torch.no_grad():
		modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
		
		model_type = config_data['training_mode']
		this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
		this_net.eval()

		lrpdm = LRP_diary_manager(config_data, this_net.training_cycle, more_text=diary_header_text)

		if mode == '0001':
			case_numbers_manual = case_numbers
			output_folder_name = 'lrp_filter_sweep_mode_0001'
			for_evaluation = generic_data_loading(config_data, case_numbers_manual=case_numbers_manual, case_type='training')
		elif mode == '0002':
			case_number_to_keep = case_numbers
			for_evaluation = generic_data_loading(config_data, case_type='training')
			output_folder_name = 'lrp_filter_sweep_mode_0002'

			import utils.evalobj as ev
			evLRP = ev.EvalLRP(config_data, output_folder_name, save_name='EvalLrp.evlrp')
			

		print("\n========== starting LRP filter sweeper: ==========")
		for case_number in for_evaluation:
			lrpdm.start_timing_per_case()

			filter_sweep_options = get_filter_sweep_options(mode='filter_sweep_0001_OPTIONS')
			print("case number:%s"%(str(case_number)))
			x, labels = for_evaluation[case_number] # shapes are like torch.Size([1, 6, 19, 192, 192]) torch.Size([256, 256, 24])	
			s, sx = labels.shape, np.array(x.shape)
			no_of_modalities = sx[1]

			R, y, outputs_label = convert_prediction_to_correct_size(this_net, x, labels)
			compute_dice_and_update_LRP_diary(lrpdm, case_number, dice_loss, outputs_label, labels)

			if mode == '0001':
				lrp_output_dictionary = build_output_dictionary(relprop_config,this_net, R, x, y, labels, sx, s, no_of_modalities, filter_sweep_options, verbose=0)
				save_one_LRP_output_dictionary(lrp_output_dictionary, case_number ,config_data, case_type='training', output_folder_name=output_folder_name)
			elif mode == '0002':
				lrp_output_dictionary = build_output_dictionary(relprop_config,this_net, R, x, y, labels, sx, s, no_of_modalities, filter_sweep_options, verbose=0)
				
				if case_number in case_number_to_keep: 
					save_one_LRP_output_dictionary(lrp_output_dictionary, case_number ,config_data, case_type='training', output_folder_name=output_folder_name)
					print('  .lrpd file SAVED')
				else: print('  .lrpd file NOT SAVED')

				evLRP.process_one_case(case_number, lrp_output_dictionary, processing_mode='0002', verbose=250)
			lrpdm.stop_timing_per_case()
		if verbose==0: print()

		if mode == '0002':
			print("========= save & review ========================")
			# evLRP.print_pfilter_data_0002()
			evLRP.save_object(evLRP.save_fullpath)
			print('Done.')

def lrp_UNet3D_overfit_options(config_data, output_lrp_folder_name='output_lrp', verbose=0):
	relprop_config = config_data['LRP']['relprop_config']
	dice_loss = uloss.SoftDiceLoss()
	with torch.no_grad():
		modalities_dict, no_of_input_channels = get_modalities_0001(config_data)
		
		model_type = config_data['training_mode']
		this_net = nut.get_UNet3D_version(config_data, no_of_input_channels, training_mode=model_type, training=False)
		this_net.eval()

		lrpdm = LRP_diary_manager(config_data, this_net.training_cycle, more_text='lrp_UNet3D_overfit_options()')
		
		for_evaluation = generic_data_loading(config_data)

		print("\n========== starting LRP evaluation: ==========")
		print("  normalization_mode:%s"%(relprop_config['normalization']))
		cases_processed = []
		for case_number in for_evaluation:
			if verbose>=90: print("case number:%s"%(str(case_number)))
			x, labels = for_evaluation[case_number] # shapes are like torch.Size([1, 6, 19, 192, 192]) torch.Size([256, 256, 24])
			s, sx = labels.shape, np.array(x.shape)
			no_of_modalities = sx[1]
			if verbose>=100: print("  s=%s,sx=%s, no_of_modalities=%s"%(str(np.array(s)),str(sx),str(no_of_modalities)))

			R, y, outputs_label = convert_prediction_to_correct_size(this_net, x, labels)
			compute_dice_and_update_LRP_diary(lrpdm, case_number, dice_loss, outputs_label, labels)

			if DEBUG_lrp_relprop_one: R = this_net.relprop_debug(R,relprop_config); break
			R = this_net.relprop(R, relprop_config).squeeze()
			Rc = convert_LRP_output_to_correct_size(R, x, sx, s, no_of_modalities, verbose=verbose)

			save_one_lrp_output(case_number,y,Rc, config_data, output_lrp_folder_name=output_lrp_folder_name)
			cases_processed.append(case_number)
	
			if verbose<=10:
				if len(cases_processed)==1: print(" progress:",end='')
				if len(cases_processed)%5==0: print("$",end='')

		print("\n  cases_processed:%s"%(cases_processed))

