import torch

def DSC(y_pred, y_true):
	'''
	Assuming input of batch size N, i.e. shape is N, w,h,d or other volume  shapes like N,d,h,w
	'''
	N = y_true.size(0)
	eps = 1e-7

	y_pred_flat = y_pred.contiguous().view(N, -1).to(torch.float)
	y_true_flat = y_true.contiguous().view(N, -1).to(torch.float)
	intersection = (y_pred_flat * y_true_flat)
	print('Inter & Union: ', intersection.sum(1).cpu().numpy(), y_pred_flat.sum(1).cpu().numpy(), y_true_flat.sum(1).cpu().numpy())

	dsc = 2 * intersection.sum(1) / (y_pred_flat.sum(1) + y_true_flat.sum(1) + eps)
	avg_dsc = dsc.sum() / N

	return avg_dsc
