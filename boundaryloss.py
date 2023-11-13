import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from myutils import onehot
def BoundaryLoss(input ,target):

    input = nn.Sigmoid()(input)
    #print(target.shape)
    #target[0,:,:,:]= target[0,:,:,:].mean(dim=1).view(1,64,64)
    #target[1, :, :, :] = target[1, :, :, :].mean(dim=1).view(1, 64, 64)
    target[target>0] = 1
    target[target<=0] = 0
    target = onehot(target)

    #print("input------",input)

    loss = - target * torch.log(input) - (1 - target) * torch.log(1 - input)
    #print("bou_loss:", loss)
    #print("loss----------",loss)
    loss = loss[target!= 0]

    #return - torch.log(input.gather(1, target.view(-1, 1))).mean()
    return loss.mean()
