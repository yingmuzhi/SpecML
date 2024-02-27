import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"
os.environ["NCCL_DEBUG"] = "INFO"
import dataset, model
from torch.utils.data import DataLoader
import torch.nn as nn, torch, random, numpy as np

def set_seed(seed: int):
    """
    intro:
        set random seed, make the final results the same.
    args:
        :param int seed: seed value.
    return:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_pth(load_path: str, net: torch.Tensor):
    """
    intro:
        load weight files.
    """
    net.load_state_dict(torch.load(load_path))

def save_pth(save_path: str, net: torch.Tensor):
    """
    intro:
        save weight files.
    """
    torch.save(net.state_dict(), save_path)
    return

def validata_data():
    file_path = "/home/yingmuzhi/SpecML/src/d2o.txt"
    with open(file_path, "r") as file:
        data_str = file.read()

    # 按照换行符分割字符串，然后将每行的浮点数转换为 Python 的 float 类型
    data_list = [float(num) for num in data_str.split('\n') if num.strip()]

    result = (torch.tensor(data_list, dtype=torch.float32), torch.tensor([0], dtype=torch.float32))
    return result

def train():
    # set seed
    seed = 3407 # ref `https://arxiv.org/abs/2109.08203v2`
    set_seed(seed)

    # dataset
    folder_path = "/home/yingmuzhi/SpecML/src/FTIR/1_data_mapping.csv"
    mydataset = dataset.one_dimension_spectrum_data(folder_path)
    train_loader = DataLoader(mydataset, batch_size=1, shuffle=False)
    print(train_loader)

    # model
    net = model.Net()
    print(net)
    # 加载预训练模型
    # save_pth_path = "/home/yingmuzhi/SpecML/src/pth/test.pth"
    # load_pth(save_pth, net)

    # train
    loss_function = nn.MSELoss()
    optimizer =  torch.optim.SGD(net.parameters(), lr = 1e-3)

    # priori test
    net.eval()
    with torch.no_grad():
            X, y = validata_data()
            X = X[0: dataset.one_dimension_spectrum_data(folder_path).CUTTING_LENGTH]
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)
            pred = net(X)
            loss = loss_function(pred, y)
            print("priori validate loss is {}".format(loss.detach().numpy()))
            pass

    for epoch in range(10):
        # train
        net.train()
        for i, (X, y) in enumerate(train_loader):
            pred = net(X)
            loss = loss_function(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch = {}, i = {}, loss = {}".format(epoch, i, loss.detach().numpy()))

            # save pth
            save_pth_path = "/home/yingmuzhi/SpecML/src/pth/test.pth"
            save_pth(save_pth_path, net)
        
        # validate
        net.eval()
        with torch.no_grad():
            X, y = validata_data()
            X = X[0: dataset.one_dimension_spectrum_data(folder_path).CUTTING_LENGTH]
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)
            pred = net(X)
            loss = loss_function(pred, y)
            print("validate loss is {}".format(loss.detach().numpy()))
            pass

if __name__ == '__main__':
    train();



