from utils.utils import *

from utils.printing_manager import PrintingManager
pm = PrintingManager()
def show_compare_other_results(dilute_result_dict):
	print('show_compare_other_results()')

	diff_set = []
	dice_score_set = []
	diff_set2 = []
	dice_score_set2 = []

	COMPARISON_THRESHOLD = 0.3

	pm.printvm('%s'%('gaussian_test_data'), tab_level=1)
	for case_number, defect_test_data in dilute_result_dict['gaussian_test_data_by_case_number'].items():
		pm.printvm('%s, %s, %s'%(str(case_number),str(type(defect_test_data)), str(len(defect_test_data))), tab_level=2) 
		for one_defect_test in defect_test_data:
			diff = one_defect_test['diff']
			dice_score = one_defect_test['dice_score']

			if dice_score>COMPARISON_THRESHOLD:
				diff_set.append(diff)
				dice_score_set.append(dice_score)

	pm.printvm('%s'%('zeroing_test_data'), tab_level=1)
	for case_number, defect_test_data in dilute_result_dict['zeroing_test_data_by_case_number'].items():
		pm.printvm('%s, %s, %s'%(str(case_number),str(type(defect_test_data)), str(len(defect_test_data))), tab_level=2) 
		for one_defect_test in defect_test_data:
			diff = one_defect_test['diff']
			dice_score = one_defect_test['dice_score']

			if dice_score>COMPARISON_THRESHOLD:
				diff_set2.append(diff)
				dice_score_set2.append(dice_score)


	show_dice_vs_diff(diff_set, dice_score_set, diff_set2, dice_score_set2)

def show_dice_vs_diff(diff_set, dice_score_set, diff_set2, dice_score_set2):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(diff_set, dice_score_set,1, c='b', label='UGauss')
	ax.scatter(diff_set2, dice_score_set2,1, c='r', label='Zero')

	ax.set_xlabel('diff')
	ax.set_ylabel('Dice')
	ax.set_xlim([0.,0.14])

	plt.legend()
	plt.show()