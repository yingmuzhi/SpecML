import pandas as pd
import torch

CUTTING_LENGTH = 1000
# set DataFrame::path
file_path = "/home/yingmuzhi/SpecML/src/FTIR/5_G-actin_final.dat"
# set DataFrame::column_name
column_name = ["wavenumber", "intensity"]

# read DataFrame::.dat files
data = pd.read_table(file_path, sep=',', names=column_name)

# show DataFrame::Math
print(data.describe())

# read lines
print(data.head())

# select data
filtered_data = data[ (data["wavenumber"] >= 1300) & (data["wavenumber"] <= 2060)]
tensor_data = torch.tensor(filtered_data["intensity"].tolist()[:CUTTING_LENGTH]).unsqueeze(0)
pass