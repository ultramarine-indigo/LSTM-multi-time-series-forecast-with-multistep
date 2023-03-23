import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# RNNs模型基类，主要是用于指定参数和cell类型
class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, device):

        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        self.device = device
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                        num_layers=self.layerNum, dropout=0.0,
                         nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                               num_layers=self.layerNum, dropout=0.0,
                               batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True, )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)


# 标准RNN模型
class RNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, device):

        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, device)

    def forward(self, x):
        device = self.device
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize , self.hiddenNum))
        h0 = h0.to(device)
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


# LSTM模型
class LSTMModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, device):
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, device)

    def forward(self, x):
        device = self.device
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)).to(device)
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)).to(device)
        rnnOutput, (hn,cn) = self.cell(x, (h0, c0))
        #print("hn0:",hn.shape)
        #hn = hn[0].view(batchSize, self.hiddenNum)
        hn = hn[-1]
        #print("hn1:", hn.shape)
        fcOutput = self.fc(hn)

        return fcOutput


# GRU模型
class GRUModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, device):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, device)

    def forward(self, x):
        device = self.device
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        h0.to(device)
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


class ResRNN_Cell(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, device):

        super(ResRNN_Cell, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.device = device

        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        # self.ht2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        device = self.device
        batchSize = x.size(0)

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        h0 = h0.to(device)
        ht = h0

        lag = x.data.size()[1]

        outputs = []

        for i in range(lag):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)

            if i == 0:
                hstart = hn
            elif i == lag - 2:
                h0 = nn.Tanh()(hn + hstart)
            else:
                if self.resDepth == 1:
                    h0 = nn.Tanh()(hn + h0)
                else:
                    if i % self.resDepth == 0:
                        h0 = nn.Tanh()(hn + ht)
                        ht = hn
                    else:
                        h0 = nn.Tanh()(hn)
            # act_hn = self.act(hn)
            outputs.append(hn)

        output_hiddens = torch.cat(outputs, 0)

        return output_hiddens


# ResRNN模型
class ResRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, device):

        super(ResRNNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.device = device

        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.ht2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)

        self.i2h = self.i2h.to(device)
        self.h2h = self.h2h.to(device)
        self.h2o = self.h2o.to(device)
        self.fc = self.fc.to(device)
        self.ht2h = self.ht2h.to(device)

    def forward(self, x):
        device = self.device
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        h0 = h0.to(device)
        lag = x.data.size()[1]
        ht = h0
        for i in range(lag):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)

            if i == 0:
                hstart = hn
            elif i == lag-1:
                h0 = nn.Tanh()(hn+hstart)
            else:
                if self.resDepth == 1:
                    h0 = nn.Tanh()(hn + h0)
                else:
                    if i % self.resDepth == 0:
                        h0 = nn.Tanh()(hn + ht)
                        ht = hn
                    else:
                        h0 = nn.Tanh()(hn)

        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput
