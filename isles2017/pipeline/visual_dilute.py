from utils.utils import *

from utils.printing_manager import PrintingManager
pm = PrintingManager()
def show_dilute_results(dilute_result_dict):
	print('show_dilute_results()')
	 
	diff_set = []
	dice_score_set = []
	diff_dilute_set_by_case_numbers = []
	dice_score_dilute_set_by_case_numbers = []

	COMPARISON_THRESHOLD = 0.3

	pm.printvm('%s'%('defect_test_data'), tab_level=1) 	
	for case_number, defect_test_data in dilute_result_dict['defect_test_data_by_case_number'].items():
		pm.printvm('%s, %s, %s'%(str(case_number),str(type(defect_test_data)), str(len(defect_test_data))), tab_level=2) 
		for one_defect_test in defect_test_data:
			diff = one_defect_test['diff']
			dice_score = one_defect_test['dice_score']

			if dice_score > COMPARISON_THRESHOLD:
				diff_set.append(diff)
				dice_score_set.append(dice_score)
			# pm.printvm('%s'%(str(dice_score),), tab_level=4) 
		

	pm.printvm('%s'%('dilute_test_data'), tab_level=1) 	
	for case_number, dilute_test_data in dilute_result_dict['dilute_test_data_by_case_number'].items():
		# print(case_number, type(dilute_test_data)) 
		pm.printvm('%s, %s, %s'%(str(case_number),str(type(defect_test_data)), str(len(dilute_test_data))), tab_level=2) 
		this_diff_set, this_dice_score_set = [], []
		for one_defect_test in dilute_test_data:
			diff = one_defect_test['diff']
			dice_score = one_defect_test['dice_score']

			if dice_score > COMPARISON_THRESHOLD:
				this_diff_set.append(diff)
				this_dice_score_set.append(dice_score)
			# pm.printvm('%s'%(str(dice_score),), tab_level=4) 
		diff_dilute_set_by_case_numbers.append(this_diff_set)
		dice_score_dilute_set_by_case_numbers.append(this_dice_score_set)

	show_dice_vs_diff(diff_set, dice_score_set, diff_dilute_set_by_case_numbers, dice_score_dilute_set_by_case_numbers)

def show_dice_vs_diff(diff_set, dice_score_set, diff_dilute_set_by_case_numbers, dice_score_dilute_set_by_case_numbers):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(diff_set, dice_score_set,1, c='b', label='diffgen')
	this_label = 'dilute'
	for diff, dice in zip(diff_dilute_set_by_case_numbers, dice_score_dilute_set_by_case_numbers):
		if len(diff) == 0: 
			continue
		ax.plot(diff, dice, c='cyan', label=this_label, linestyle='--', marker='x', markeredgecolor='r')

		this_label = None
		
	ax.set_xlabel('diff')
	ax.set_ylabel('Dice')

	plt.legend()
	plt.show()

