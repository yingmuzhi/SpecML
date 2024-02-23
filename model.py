import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, input=28*28, output=28*28):
        super().__init__()
        # define any number of nn.Modules (or use your current ones)
        self.prior = nn.Sequential(nn.Linear(1000, 28 * 28))
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
        self.rear = nn.Sequential(nn.Linear(28 * 28, 14 * 14), nn.ReLU(), nn.Linear(14 * 14, 7 * 7), nn.ReLU(), nn.Linear(7 * 7, 1))
    def  forward(self, input):
        x = self.prior(input)
        y = self.encoder(x)
        z = self.decoder(y)
        output = self.rear(z)
        return output
Net = TinyNet
if __name__=="__main__":
    net = Net()
    x = torch.randn((1000))
    z = net(x)
    pass
