from utils.utils import *

class SoftDiceLoss(nn.Module):
    """
    sdl = SoftDiceLoss()
    sdl.smooth=1.
    logits = torch.Tensor([[1., 1., 0.],[1., 1., 0.]])
    targets = torch.Tensor([[1., 1., 0.],[1., 1., 0.]])
    targets2 = torch.Tensor([[0., 1., 0.],[1., 1., 0.]])
    targets3 = torch.Tensor([[0., 1., 1.],[1., 1., 0.]])
    targets4 = torch.Tensor([[0., 0., 1.],[0., 0., 1.]])
    t = [targets, targets2, targets3, targets4]
    for tt in t:
        dice = sdl(logits, tt, factor=1).item()
        print(dice)

    targetsALLZERO = torch.Tensor([[0., 0., 0.],[0., 0., 0.]])
    dice = sdl(logits, targetsALLZERO, factor=1).item()
    print("dice target all zero:", dice)
    dice = sdl(targetsALLZERO, targetsALLZERO, factor=1).item()
    print("dice both zeros", dice)

    Output:
    -1.0
    -0.875
    -0.7777777910232544
    -0.1428571492433548
    dice target all zero: -0.20000000298023224
    dice both zeros -1.0
    """
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
        self.smooth = 1
    def forward(self, logits, targets, factor=1, print_info=None):        
        num = targets.numel()
        m1 = logits.contiguous().view(-1).to(torch.float)
        m2 = targets.contiguous().view(-1).to(torch.float)
        # print(" .SoftDiceLoss().",m1.shape,m2.shape)
        if not m1.shape == m2.shape:
            raise Exception('SoftDiceLoss. forward(). Shapes do not match.') 
        intersection = m1 * m2
        if print_info is not None:
            if print_info >= 19:
                print("   m1.sum(), m2.sum()       = %d, %d\n   intersection.sum(), num =  %d, %d" % (m1.sum(), m2.sum(), intersection.sum(), num))

        score = (2. * intersection.sum() + self.smooth) / (m1.sum() + m2.sum() + self.smooth)
        score = (1 - score) * factor
        
        '''DEBUG'''
        # score = torch.Tensor(np.random.normal(0.5,1, size=(1,)))
        return score

class CustomPrecision(nn.Module):
    def __init__(self):
        super(CustomPrecision, self).__init__()
        self.smooth = 1

    def forward(self, logits, targets, print_info=None): 
        m1 = logits.view(-1).to(torch.float)
        m2 = targets.view(-1).to(torch.float)
        all_P = m1.sum()
        TP = (m1 * m2).sum()
        precision = (TP+self.smooth) / (all_P+self.smooth)
        return precision

class CustomRecall(nn.Module):
    def __init__(self):
        super(CustomRecall, self).__init__()
        self.smooth = 1
        
    def forward(self, logits, targets, print_info=None): 
        m1 = logits.view(-1).to(torch.float)
        m2 = targets.view(-1).to(torch.float)
        FN = ((m1-m2)<0).to(torch.float).sum()
        TP = (m1 * m2).sum()
        recall = (TP+self.smooth) / (TP + FN+self.smooth)
        return recall      