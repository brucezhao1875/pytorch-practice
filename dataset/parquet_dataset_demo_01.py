'''
wmt dataset 2500万的句子对，需要怎么进行dataset操作才可以维持效率
差不多一个分区的话，遍历1遍耗时132秒。

实践发现：01、07 文件读取失败，可能是文件格式被破坏。检查md5值
windows：CertUtil
linux：md5sum
'''
import pandas as pd
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#FILE_PATH = 'e:/PycharmProjects/wmt-dataset/test-00000-of-00001.parquet'
FILE_PATH = 'e:/PycharmProjects/wmt-dataset/train-00012-of-00013.parquet'
df = pd.read_parquet(FILE_PATH)

class ParquetDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        # 直接获取字典
        translation_dict = self.dataframe.iloc[idx].to_dict()

        # 现在 translation_dict 应该是 {'translation': {'en': '英文文本', 'zh': '中文文本'}}
        # 我们需要获取 'translation' 键对应的值
        translation_dict = translation_dict['translation']

        # 提取 'en' 和 'zh' 键对应的文本
        en_text = translation_dict['en']
        zh_text = translation_dict['zh']

        # 将文本转换为张量
        en_tensor = self.text_to_tensor(en_text)
        zh_tensor = self.text_to_tensor(zh_text)

        return en_tensor, zh_tensor
    def text_to_tensor(self, text):
        # 这里是将文本转换为张量的示例实现，你需要根据你的任务进行实现
        # 例如，使用分词器、构建索引、转换为张量等
        # 这里返回None作为占位符
        return text


#use ParquetDataset
dataset = ParquetDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

time_start = time.time()
#遍历dataloader
for batch_idx, data in enumerate(dataloader):
    if batch_idx == 0:
        en_list,zh_list = data

    inputs, targets = data
    if batch_idx % 1000 == 0:
        print(f"batch {batch_idx} dealed.")
time_end = time.time()
print(f"one epoch train done in {time_end-time_start} seconds.")
