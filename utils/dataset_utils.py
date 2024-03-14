import pandas as pd, torch
import os 

def generate_dataset_csv(folder_path: str, save_path: str=None):
    """
    intro:
        generate data csv file.
    args:
        :param str folder_path: where data store.
        :param str save_path: save csv file path.
    return:
        :param str save_path:
    """
    data_mapping = {}
    if not save_path:
        save_path = os.path.join(folder_path, "1_data_mapping.csv")


    for target in os.listdir(folder_path):
        target_path = os.path.join(folder_path, target)

        # find wether is directory or file
        if os.path.isdir(target_path):
            # traversal files in directory
            for signal_name in os.listdir(target_path):
                signal_path = os.path.join(target_path, signal_name)

                # mapping
                data_mapping[signal_path] = target

    df = pd.DataFrame(list(data_mapping.items()), columns=["signal_path", "target_value"])

    # save as .csv file
    df.to_csv(save_path, index=False)
    print("Data and label mapping saved to 'data_mapping.csv'")
    return save_path 


def read_one_signal(file_path: str, CUTTING_LENGTH: int=1000):
    """
    intro:
        read one 1D signal
    args:
        :param str file_path: the file path to read data.
        :param int CUTTING_LENGTH: the length of data u want read from file.
    return:
        :param torch.Tensor: shape is [channel, 1D data]
    """
    # set column name
    column_name = ["wavenumber", "intensity"]

    # read DataFrame::.dat files
    data = pd.read_table(file_path, sep=',', names=column_name)

    # show DataFrame::Math
    print(data.describe())
    # read lines
    print(data.head())

    # select data
    filtered_data_main = data[ ((data["wavenumber"] >= 1300) & (data["wavenumber"] <= 2060)) ][150: 850]    # get protein backbone information   
    filtered_data_side = data[ ((data["wavenumber"] >= 2780) & (data["wavenumber"] <= 3050)) ][50: 350]     # get protein side chain
    filtered_data = pd.concat([filtered_data_main, filtered_data_side])
    filtered_tensor = torch.tensor(filtered_data["intensity"].tolist()[:CUTTING_LENGTH], 
                                   dtype=torch.float32) 
    
    # 缺少的地方用0填充
    temp_tensor = torch.zeros((CUTTING_LENGTH)) 
    temp_tensor[ :filtered_tensor.shape[0]] = filtered_tensor

    tensor_data = torch.tensor(temp_tensor, 
                               dtype=torch.float32).unsqueeze(0)

    return tensor_data


def read_one_target(target_value: int):
    """
    intro:
        return target value.
    args:
        :param int target_value:
    return:
        :param torch.Tensor target:
    """
    target = torch.tensor(target_value, dtype=torch.float32).unsqueeze(0)
    return target


if __name__=="__main__":
    file_path = "/home/yingmuzhi/SpecML/src/FTIR/5_G-actin_final.dat"
    data = read_one_signal(file_path)

    generate_dataset_csv(folder_path="/home/yingmuzhi/SpecML/src/FTIR")
    pass