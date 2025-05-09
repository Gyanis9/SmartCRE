import json
from typing import List, Dict, Tuple, Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Processor:
    """处理 TACRED/FewRel 数据集的类，包含数据加载、清洗及划分功能"""

    @staticmethod
    def load_data(file_path: str) -> Dict[str, List[Dict]]:
        """从JSON文件加载原始数据"""

        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def divided_data(config) -> tuple[list[Any], dict[Any, Any], dict[Any, Any]]:
        """划分数据集为训练/验证/测试集"""
        raw_data = Processor.load_data(config.data_file)

        train_data = {}
        dev_data = {}
        test_data = {}
        all_relations = set()

        # 遍历每个关系类型的数据
        for relation, samples in tqdm(raw_data.items(), desc="Processing relations"):
            if config.task_name == 'FewRel':
                # FewRel 数据集划分逻辑
                test_split = config.num_of_train_samples + config.num_of_val_samples
                train_samples = samples[:config.num_of_train_samples]
                test_samples = samples[test_split:]
            else:
                # TACRED 默认划分逻辑
                test_size = min(len(samples) // 5, 40)
                test_samples = samples[:test_size]
                train_samples = samples[test_size:test_size + 320]
                dev_samples = None

            train_data[relation] = train_samples
            test_data[relation] = test_samples
            all_relations.add(relation)

        return list(all_relations), train_data, test_data


class RelationDataset(Dataset):
    """数据集类，用于处理 数据集，基于 PyTorch 的 Dataset 类进行封装"""

    def __init__(self, data, tokenizer, relation_to_id: Dict[str, int]):
        """初始化 Dataset 类"""

        self.data = data
        self.tokenizer = tokenizer
        self.relation_to_id = relation_to_id

    def __len__(self) -> int:
        """返回数据集的大小，即数据样本的数量"""

        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据集中指定索引的数据样本，进行处理并返回"""

        item = self.data[idx]
        encoding = self.tokenizer(
            ' '.join(item['tokens']),
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.relation_to_id[item['relation']], dtype=torch.long)
        }
