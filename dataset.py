from pyparsing import Any
from torch.utils.data import DataLoader
# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import pandas as pd
import utils


class one_dimension_spectrum_data(Dataset):
    def __init__(self, data_csv_path, CUTTING_LENGTH = 1000) -> None:
        super().__init__()
        df_data = pd.read_csv(data_csv_path)
        self.CUTTING_LENGTH = CUTTING_LENGTH
        self.signal = df_data.loc[:, "signal_path"].tolist()
        self.target = df_data.loc[:, "target_value"].tolist()

    def __getitem__(self, index) -> Any:
        signal = utils.dataset_utils.read_one_signal(self.signal[index], CUTTING_LENGTH=self.CUTTING_LENGTH)
        target = utils.dataset_utils.read_one_target(self.target[index])
        return signal, target
    
    def __len__(self):
        return len(self.target)


if __name__ == '__main__':
    folder_path="/home/yingmuzhi/SpecML/src/FTIR"
    data_csv_path = utils.dataset_utils.generate_dataset_csv(folder_path)

    mydataset = one_dimension_spectrum_data(data_csv_path)
    print(mydataset[1])
    train_loader = DataLoader(mydataset, batch_size=1, shuffle=True)
    print(next(iter(train_loader)))
    pass