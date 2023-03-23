import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



def predict(net, x, device):

    net.to(device)
    net = net.eval()
    x = torch.Tensor(x).to(device)
    pred = net(x)
    #pred = pred.data.numpy()
    return pred


def predict_iteration(net, x, look_ahead, device):

    #batch_size = X.shape[0]
    ans = []
    input = torch.Tensor(x).to(device)

    for i in range(look_ahead):
        pred = net(input)
        ans.append(pred)
        input = torch.cat([input,pred],1)
        input = input[:, 1:, :]  # drop the head

    ans = np.array(ans)
    return ans