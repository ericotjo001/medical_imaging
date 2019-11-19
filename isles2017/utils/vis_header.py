def rewrap_filter_sweep_dictionary(filtermode, output_dictionary, verbose=0):
	"""
	Ad hoc function for vis_filter_sweeper()
	We want to extract both filtermode
	1. fraction_pass_filter
	2. fraction_clamp_filter

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

	"""
	graded_output = []
	graded_output_label = []
	if verbose>=100:
		print('rewrap_filter_sweep_dictionary()')
		print('  filtermode:%s'%(filtermode))
	for xkey in output_dictionary:
		# print(xkey[0], filtermode==xkey[0])
		if filtermode==xkey[0]:
			filter_label = str(xkey[1])+','+str(str(xkey[2]))
			if verbose>=100: print("    ",filter_label,output_dictionary[xkey].shape)
			graded_output.append(output_dictionary[xkey])
			graded_output_label.append(filter_label)
	return graded_output_label, graded_output