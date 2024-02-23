from pyparsing import Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch

CUTTING_LENGTH = 1000
file_path_list = ["/home/yingmuzhi/SpecML/src/g-actin.txt", "/home/yingmuzhi/SpecML/src/f-actin.txt", "/home/yingmuzhi/SpecML/src/d2o.txt"]
def read_data(file_path: str):
    with open(file_path, "r") as file:
        data_str = file.read()

    # 按照换行符分割字符串，然后将每行的浮点数转换为 Python 的 float 类型
    data_list = [float(num) for num in data_str.split('\n') if num.strip()]
    return data_list

g_actin = read_data(file_path_list[0])
f_actin = read_data(file_path_list[1])
d2o = read_data(file_path_list[2])
train_signal = torch.tensor([
    g_actin,
    f_actin
], dtype = torch.float32)

train_target = torch.tensor([
    [0], 
    [1],
], dtype = torch.float32)


class one_dimension_spectrum_data(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.input = train_signal; self.output = train_target

    def __getitem__(self, index) -> Any:
        return self.input[index][:CUTTING_LENGTH], self.output[index][:CUTTING_LENGTH]
    
    def __len__(self):
        return len(self.output)
    
if __name__ == '__main__':
    mydataset = one_dimension_spectrum_data()
    print(mydataset[1])
    train_loader = DataLoader(mydataset, batch_size=1, shuffle=True)
    pass