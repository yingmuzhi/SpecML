import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"
os.environ["NCCL_DEBUG"] = "INFO"
import dataset, model, utils
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

def train():
    # set seed
    seed = 3407 # ref `https://arxiv.org/abs/2109.08203v2`
    set_seed(seed)

    # dataset
    folder_path = "/home/yingmuzhi/SpecML/src/Data/FTIR/1_data_mapping.csv"
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
        file_path = "/home/yingmuzhi/SpecML/src/Data/FTIR/Insulin+0.2MCu.10.dat"
        X = utils.dataset_utils.read_one_signal(file_path)
        y = torch.tensor([0], dtype=torch.float32).unsqueeze(0)
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
            save_pth_path = "/home/yingmuzhi/SpecML/src/pth/test_FTIR.pth"
            save_pth(save_pth_path, net)
        
        # validate
        net.eval()
        with torch.no_grad():
            file_path = "/home/yingmuzhi/SpecML/src/Data/FTIR/Insulin+0.2MCu.10.dat"
            X = utils.dataset_utils.read_one_signal(file_path)
            y = torch.tensor([0], dtype=torch.float32).unsqueeze(0)
            pred = net(X)
            loss = loss_function(pred, y)
            print("priori validate loss is {}".format(loss.detach().numpy()))
            pass

if __name__ == '__main__':
    train();



