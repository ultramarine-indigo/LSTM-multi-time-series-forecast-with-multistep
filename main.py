import torch
import numpy as np
import argparse
import time
from predict import *
from solver import *
parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--use_cuda',type=bool,default=False,help='')

parser.add_argument('--data',type=str,default='CDR')
parser.add_argument('--data_path',type=str,default='E:/py_projects/GDN/data/CDR/result.csv',help='data path')
parser.add_argument('--feature',type=int, default=10,help='')
parser.add_argument('--look_back',type=int,default=12,help='look back timestamp')
parser.add_argument('--predict_step',type=int,default=6,help='')

parser.add_argument('--mode', type=str, default='train', help='train,test')
parser.add_argument('--epoch',type=int, default=10,help='epochs')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--check_point', type=int, default=10, help='print loss iter')

parser.add_argument('--model',type=str,default='LSTM',help='LSTM,RNN,GRU')
parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--hidden_num',type=int, default=128,help='')

parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')

parser.add_argument('--rnn_flag',type=bool,default=False,help='')


config = parser.parse_args(args=[])
#args = vars(config)
args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("device:", device)
    #data = load_data(filename=args.data_path, feature=args.feature)
    # training and testing
    #testPred, testY = single_step_model_forecasting(data_path=args.data_path, feature=args.feature, look_back=args.look_back,
    #                                            epoch=args.epoch, lr=args.learning_rate,  batch_size=args.batch_size, method=args.model,
    #                                            hidden_num=args.hidden_num, layers=args.layers, check_point=args.check_point,device=device)

    testPred, testY = multi_step_model_forecasting(data_path=args.data_path, feature=args.feature,look_back=args.look_back, steps=args.predict_step,
                                                    epoch=args.epoch, lr=args.learning_rate, batch_size=args.batch_size, check_point=args.check_point,
                                                    method=args.model, hidden_num=args.hidden_num, layers=args.layers, device=device)

