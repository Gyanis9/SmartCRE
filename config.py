import argparse
from typing import Any


def get_config() -> Any:
    """获取并解析配置参数"""
    parser = argparse.ArgumentParser()

    # ================================ 基础设置 ========================================
    parser.add_argument('--experiment_name', default=None, type=str, help="实验名称，用于区分不同的实验")
    parser.add_argument('--total_rounds', default=5, type=int, help="总的训练轮次")
    parser.add_argument('--random_seed', default=2025, type=int, help="随机种子，确保实验可复现")
    parser.add_argument('--task_name', default='TACRED', type=str, help="任务名称，例如 TACRED 数据集")
    parser.add_argument('--device', default='cuda', type=str, help="训练设备，支持 'cuda' 或 'cpu'")

    # ================================ 数据集设置 ======================================
    parser.add_argument('--num_of_relations', default=40, type=int, help="数据集中总的关系数量")
    parser.add_argument('--max_sequence_length', default=256, type=int, help="输入序列的最大长度，超过该长度会被截断")
    parser.add_argument('--relations_per_task', default=4, type=int, help="每个任务中包含的关系数量")
    parser.add_argument('--num_of_train_samples', default=420, type=int, help="训练集样本数量")
    parser.add_argument('--num_of_val_samples', default=140, type=int, help="验证集样本数量")
    parser.add_argument('--num_of_test_samples', default=140, type=int, help="测试集样本数量")
    parser.add_argument('--data_file', default='./data/Data_TACRED.json', type=str, help="数据集文件路径")
    parser.add_argument('--cache_file', default='./data/Cache_TACRED.pt', type=str,
                        help="缓存文件路径，用于加速数据加载")
    parser.add_argument('--relation_file', default='./data/id2rel_TACRED.json', type=str, help="关系映射文件路径")
    parser.add_argument('--batch_size', default=32, type=int, help="每步训练的批处理大小")

    # ================================ 模型设置 ========================================
    parser.add_argument('--bert_model_path', default='bert-base-uncased', type=str, help="BERT模型的预训练路径")
    parser.add_argument('--dropout_rate', default=0.1, type=float, help="Dropout比率，用于防止过拟合")
    parser.add_argument('--encoding_mode', default='entity_marker', type=str, help="实体标记方式，可选 'entity_marker'")

    # ================================ 训练参数设置 ================================
    parser.add_argument('--max_grad_norm', default=10, type=int, help="最大梯度裁剪阈值，用于防止梯度爆炸")
    parser.add_argument("--encoder_lr", default=1e-5, type=float, help="编码器的初始学习率，用于AdamW优化器")
    parser.add_argument("--classifier_lr", default=1e-3, type=float, help="分类器的初始学习率，用于AdamW优化器")

    # ================================ 优化器设置 ============================
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Adam优化器的epsilon参数")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="权重衰减，防止过拟合")

    # ================================ 记忆容量设置 ============================
    parser.add_argument('--memory_size', default=10, type=int, help="每个任务的记忆大小，控制每个任务保存的信息量")
    parser.add_argument("--min_memory", default=5, type=int, help="每个任务中最少保留的记忆数量")
    parser.add_argument("--max_memory", default=20, type=int, help="每个任务中最多保留的记忆数量")
    parser.add_argument("--sensitivity", default=5, type=int, help="影响记忆更新频率的灵敏度参数")

    # ================================ 损失函数训练参数设置 ============================

    # 解析并返回配置
    config = parser.parse_args()
    return config
