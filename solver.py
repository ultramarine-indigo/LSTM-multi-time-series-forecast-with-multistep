from sklearn.preprocessing import MinMaxScaler
from predict import predict,predict_iteration
from train import trainer
from data_loader import *
import eval



# forecasting with sigle-step
def single_step_model_forecasting(data_path, feature, look_back, lr, epoch, batch_size, hidden_num, layers, method, device,check_point):

    # normalize time series
    dataset = load_data(filename=data_path, feature=feature)
    scaler = StandardScaler(dataset)
    data = scaler.transform(dataset)
    train, val, test = split_data(data)
    x_test, y_test=create_samples(test,look_back)

    #print("train X shape:", x_train.shape)
    #print("train y shape:", y_train.shape)
    #print("val X shape:", x_val.shape)
    #print("val y shape:", y_val.shape)
    print("test X shape:", x_test.shape)
    print("test y shape:", y_test.shape)



    net = trainer(data_path=data_path, num_epoch=epoch, lr=lr, batch_size=batch_size,feature=feature, checkpoint=check_point,
                look_back=look_back, method=method, hidden_num=hidden_num,layernum=layers, device=device)

    testPred = predict(net,x_test,device)
    #testPred = predict_iteration(net, testX, steps, use_cuda=use_cuda, RNN=flag)
    # trainPred = predict_iteration(net, trainX, h_train, use_cuda=use_cuda, RNN=flag)
    # print("train pred shape:", trainPred.shape)
    print("test pred shape:", testPred.shape)

    testPred = testPred.cpu().detach().numpy()
    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(y_test)



    # evaluation
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, testY


# forecasting with sigle-step
def multi_step_model_forecasting(data_path, feature, look_back, steps , lr, epoch, batch_size, hidden_num, layers, method, check_point,device):

    # normalize time series

    dataset = load_data(filename=data_path, feature=feature)
    scaler = StandardScaler(dataset)
    data = scaler.transform(dataset)

    train, val, test = split_data(data)

    x_test, y_test = create_multi_ahead_samples(test, look_back, steps)

    #print("train X shape:", x_train.shape)
    #print("train y shape:", y_train.shape)
    print("test X shape:", x_test.shape)
    print("test y shape:", y_test.shape)
    #print("val X shape:", x_val.shape)
    #print("val y shape:", y_val.shape)


    net = trainer(data_path,  num_epoch=epoch, lr=lr, batch_size=batch_size,feature=feature,look_back=look_back,
                 method=method, hidden_num=hidden_num,layernum=layers, checkpoint=check_point,device=device)

    testPred = predict_iteration(net, x_test, steps, device)

    print("test pred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(y_test)

    # evaluation
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, testY


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)