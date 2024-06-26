import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
#from transformers import BertTokenizer
import time
from multiprocessing import Pool, cpu_count

'''
在demo_02的基础上做修改，尝试提升效率。
demo_02在v100服务器上：0-9分区共1900万条数据，只是enumerate(dataloader)一遍，需要1157秒。想办法提升效率。


'''

class ParquetDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_length=512):
        # 批量读取所有 Parquet 文件
        self.data = pd.concat([pd.read_parquet(file_path) for file_path in file_paths])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = [None] * len(self.data)  # 初始化缓存

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache[idx] is None:  # 如果缓存中没有，则进行处理
            translation_dict = self.data.iloc[idx].to_dict()
            en_text = translation_dict['translation']['en']
            zh_text = translation_dict['translation']['zh']
            en_tensor = self.tokenize_text(en_text)
            zh_tensor = self.tokenize_text(zh_text)
            self.cache[idx] = (en_tensor, zh_tensor)  # 缓存处理结果
        return self.cache[idx]

    def tokenize_text(self, text):
        return text
        #return self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True,
        #                      return_tensors="pt").input_ids


def collate_fn(batch):
    en_tensors, zh_tensors = zip(*batch)
    #en_tensors = torch.cat(en_tensors, dim=0)
    #zh_tensors = torch.cat(zh_tensors, dim=0)
    return en_tensors, zh_tensors


def load_data_in_parallel(dataset, num_workers):
    with Pool(num_workers) as p:
        data = p.map(dataset.__getitem__, range(len(dataset)))
    return data


def main():
    parquet_dir = 'e:/PycharmProjects/wmt-dataset'
    #parquet_dir = '/root/llm_gpt/wmt-dataset'
    #file_paths = list(Path(parquet_dir).glob('test-00000-of-00001.parquet'))
    file_paths = list(Path(parquet_dir).glob('train-0000[0-9]-of-00013.parquet'))
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = None
    dataset = ParquetDataset(file_paths, tokenizer)

    #batch_size = 512
    batch_size = 32
    num_workers = min(cpu_count(), len(file_paths))  # 并行处理的工作线程数
    print(f"working with {num_workers} num_workers...")

    # 并行预加载数据
    start_time = time.time()
    print(f"Begin Data preloading...")
    preloaded_data = load_data_in_parallel(dataset, num_workers)
    dataset.cache = preloaded_data
    preloading_time = time.time() - start_time
    print(f"Data preloading time: {preloading_time:.2f} seconds")

    # 使用 DataLoader 进行批量数据加载
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=num_workers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 示例训练循环
    for epoch in range(1):
        epoch_start_time = time.time()
        for batch_idx, (en_tensors, zh_tensors) in enumerate(dataloader):
            # 在这里进行训练代码
            if batch_idx % 1000 == 0:
                print(f"batch {batch_idx} dealed.")
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} time: {epoch_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == '__main__':
    main()
