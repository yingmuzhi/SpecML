import os

# 定义包含数据和标签的字典
data_label_mapping = {}

# 指定包含数据的文件夹路径
folder_path = '/home/yingmuzhi/SpecML/src/FTIR'  # 替换为实际的文件夹路径

# 遍历文件夹
for label in os.listdir(folder_path):
    label_path = os.path.join(folder_path, label)
    
    # 检查是否是文件夹
    if os.path.isdir(label_path):
        # 遍历文件夹中的文件
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            
            # 添加数据和标签的映射关系
            data_label_mapping[file_path] = label

# 打印数据和标签的映射关系
for data, label in data_label_mapping.items():
    print(f"Data: {data}, Label: {label}")

pass