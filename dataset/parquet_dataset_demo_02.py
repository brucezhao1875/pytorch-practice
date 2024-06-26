'''
在demo_01的基础上做修改：
1、多个分区，全部遍历
2、其它（包括shuffle等）

'''
import pandas as pd
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

class MultiFileParquetDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.dataframes = [pd.read_parquet(path) for path in file_paths]
        self.cumulative_sizes = self._calculate_cumulative_sizes()

    def _calculate_cumulative_sizes(self):
        # 计算累积大小，用于多文件索引
        sizes = [len(df) for df in self.dataframes]
        cumulative_sizes = []
        cumulative_size = 0
        for size in sizes:
            cumulative_size += size
            cumulative_sizes.append(cumulative_size)
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]  # 返回最后一个累积大小

    def __getitem__(self, idx):
        # 确定idx属于哪个DataFrame
        if idx < 0 or idx >= self.cumulative_sizes[-1]:
            raise IndexError("Index out of bounds")

        # 找到索引idx所在的DataFrame
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                df = self.dataframes[i]
                break
        else:
            df = self.dataframes[-1]

        # 计算DataFrame内的相对索引
        relative_idx = idx - (self.cumulative_sizes[i - 1] if i > 0 else 0)

        # 获取字典数据
        translation_dict = df.iloc[relative_idx].to_dict()

        # 提取 'en' 和 'zh' 键对应的文本
        en_text = translation_dict['translation']['en']
        zh_text = translation_dict['translation']['zh']

        # 将文本转换为张量
        en_tensor = self.text_to_tensor(en_text)
        zh_tensor = self.text_to_tensor(zh_text)

        return en_tensor, zh_tensor

    def text_to_tensor(self, text):
        # 这里是将文本转换为张量的示例实现，你需要根据你的任务进行实现
        # 例如，使用分词器、构建索引、转换为张量等
        # 这里返回None作为占位符
        return text

# 使用示例
# 假设 Parquet 文件都存放在同一个目录下
parquet_dir = 'e:/PycharmProjects/wmt-dataset'
parquet_files = list(Path(parquet_dir).glob('train-0000[0-9]-of-00013.parquet'))
# 创建数据集实例
dataset = MultiFileParquetDataset(parquet_files)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

time_start = time.time()
#遍历dataloader
for batch_idx, data in enumerate(dataloader):
    #if batch_idx == 0:
    #    en_list,zh_list = data

    inputs, targets = data
    if batch_idx % 1000 == 0:
        print(f"batch {batch_idx} dealed.")
time_end = time.time()
print(f"one epoch train done in {time_end-time_start} seconds.")
