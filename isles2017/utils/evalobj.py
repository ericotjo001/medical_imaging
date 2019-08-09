from utils.utils import *


class EvalObj(object):
	def __init__(self):
		super(EvalObj, self).__init__()
		self.scores = {}
		'''
		Arrange by epochs:
		self.scores = { n_epoch : {}}

		example:
		self.scores = {
			1: {
				'dice_list' : average_dice,	
			},
			2: { ... },
			3: { ... },
			...
		}

		'''
		
	def get_dice_score_at_epoch(self, epoch, dice_list):
		if epoch not in self.scores: self.scores[epoch] = {}
		self.scores[epoch]['dice_list'] = dice_list

	def save_evaluation(self, config_data):
		print('save_evaluation()')
		report_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		report_full_path = os.path.join(report_dir,'report.txt')

		txt = open(report_full_path, 'w')
		print(" %5s | %s"%("epoch", "dice_score"))
		txt.write(" %5s | %s\n"%("epoch", "dice_score"))
		for epoch in self.scores:
			avg_dice_score = np.mean(self.scores[epoch]['dice_list'])
			print(" %5s | %s"%(int(epoch), str(avg_dice_score)))
			txt.write(" %5s | %s\n"%(int(epoch), str(avg_dice_score)))
		txt.close()