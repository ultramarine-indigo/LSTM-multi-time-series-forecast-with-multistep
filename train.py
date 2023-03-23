import time
from model import *
from data_loader import get_data_loader
import torch
import numpy as np
import os
import eval


#在测试集上的损失连续几个epochs不再降低的时候，提前停止
#val_loss: 测试集/验证集上这个epoch的损失
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience  # patience: 连续patience个epochs上损失不再降低的时候，停止迭代
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2



def trainer(data_path, feature, look_back, lr, method, device, hidden_num, layernum, num_epoch, batch_size, checkpoint):


    # build up data loader

    dataloader = get_data_loader(data_path, feature, look_back, batch_size, mode='train', predict_step=1)
    net = None
    if method == "RNN":
        net = RNNModel(inputDim=feature, hiddenNum=hidden_num, outputDim=feature, layerNum=layernum, cell="RNN", device=device)
    if method == "LSTM":
        net = LSTMModel(inputDim=feature, hiddenNum=hidden_num, outputDim=feature, layerNum=layernum, cell="LSTM", device=device)
    if method == "GRU":
        net = GRUModel(inputDim=feature, hiddenNum=hidden_num, outputDim=feature, layerNum=layernum, cell="GRU", device=device)
    if method == "ResRNN":
        net = ResRNNModel(inputDim=feature, hiddenNum=hidden_num, outputDim=feature, resDepth=4, device=device)
    # if method == "attention":
    #     net = RNN_Attention(inputDim=1, hiddenNum=hidden_num, outputDim=1, resDepth=4,
    #                         seq_len=lag, merge="concate", use_cuda=use_cuda)

    net.to(device)
    net = net.train()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()    #MSE误差
    #criterion = nn.L1Loss()   # MAE误差


    #early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=data)
    train_steps = len(dataloader)
    print("data loader num:", train_steps)

    print("start training...", flush=True)
    time_now = time.time()
    for epoch in range(num_epoch):

        iter_count = 0
        train_loss=[]
        train_mae=[]
        train_rmse=[]
        train_mape=[]
        train_smape=[]
        t0 = time.time()
        for batch_idx, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()  # 参数梯度置0
            iter_count = iter_count + 1

            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            #x, y = Variable(x), Variable(y)
            #x = x.to(device)
            #y = y.to(device)


            pred = net.forward(x).to(device)
            pred = dataloader.dataset.scaler.inverse_transform(pred)
            y = dataloader.dataset.scaler.inverse_transform(y)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()


            pred = pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            train_loss.append(loss.item())
            train_mae.append(eval.calcMAE(y, pred))
            train_rmse.append(eval.calcRMSE(y, pred))
            train_mape.append(eval.calcMAPE(y, pred))
            train_smape.append(eval.calcSMAPE(y, pred))

            if batch_idx % checkpoint==0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f},Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(batch_idx, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]), flush=True)

                speed = (time.time() - t0) / iter_count
                left_time = speed * ((num_epoch - epoch) * train_steps - batch_idx)
                print('speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

        t1 = time.time()
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_smape = np.mean(train_smape)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train SMAPE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(epoch+1, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse ,mtrain_smape, (t1 - t0)),flush=True)

    t2 = time.time()
    print("Train finish. total train time:", t2-t0)

    return net




