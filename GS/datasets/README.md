# 数据集模块 (Datasets Module)

## 模块功能

数据集模块负责处理图数据集的加载、预处理和验证，支持多种类型的图数据集，包括传统引文网络、社交网络、生物信息网络和大规模OGB基准数据集。

## 提供的接口

### DatasetLoader

数据集加载器类，提供以下主要方法：

- `load_dataset(dataset_name, task_type='original', normalize_features=True)`: 加载指定数据集和任务类型
- `preprocess_for_summarization(data, remove_self_loops=True)`: 为图总结任务预处理数据
- `verify_sparse_format(data)`: 验证数据是否为有效的稀疏格式
- `load_all_datasets(task_type='original', normalize_features=True)`: 批量加载所有支持的数据集
- `get_dataset_scale(dataset_name)`: 获取数据集规模分类
- `list_datasets_by_scale()`: 按规模分类列出所有数据集

## 支持的数据集

### 传统图数据集
- **Cora**: 引文网络，2708个节点，10556条边
- **CiteSeer**: 引文网络，3327个节点，9104条边
- **PubMed**: 引文网络，19717个节点，88648条边
- **KarateClub**: 社交网络，34个节点，156条边
- **IMDB**: 社交网络，影视数据
- **PPI**: 蛋白质相互作用网络，生物信息数据

### OGB节点分类数据集

#### 小规模数据集 (Small Scale)
- **ogbn-arxiv**: ArXiv论文网络，~170K节点，学术引文数据

#### 中等规模数据集 (Medium Scale)
- **ogbn-products**: 亚马逊产品网络，~2.4M节点，电商数据
- **ogbn-proteins**: 蛋白质功能预测，~133K节点，生物数据

#### 大规模数据集 (Large Scale)
- **ogbn-mag**: 微软学术图，~1.9M节点，多类型学术数据
- **ogbn-papers100M**: 大规模论文网络，~111M节点，超大规模数据集

### 自定义数据集
- **SO_relation_ME**: 代谢网络，KO共现数据
- **SO_relation_MT**: 代谢网络，KO共现数据

## 与其它模块的对接

- **models模块**: 为图总结模型和下游任务模型提供标准化的数据格式
- **benchmark模块**: 为基准测试提供数据集加载服务
- **metrics模块**: 提供验证数据格式的工具

## 标签任务类型

每个数据集支持两种下游任务：
1. **original**: 使用数据集原始节点标签（如果可用）
2. **degree**: 基于节点度数的三分类标签（高/中/低度数）

## 使用示例

```python
from GS.datasets import DatasetLoader

# 初始化加载器
loader = DatasetLoader('./data')

# 加载传统数据集
graph, train_mask, val_mask, test_mask = loader.load_dataset('Cora', task_type='original')

# 加载OGB小规模数据集
graph, train_mask, val_mask, test_mask = loader.load_dataset('ogbn-arxiv', task_type='original')

# 加载度数标签任务
graph, train_mask, val_mask, test_mask = loader.load_dataset('Cora', task_type='degree')

# 获取数据集规模信息
scale = loader.get_dataset_scale('ogbn-products')  # 返回 'Medium'

# 按规模列出所有数据集
datasets_by_scale = loader.list_datasets_by_scale()
print(f"Small scale datasets: {datasets_by_scale['Small']}")
print(f"Medium scale datasets: {datasets_by_scale['Medium']}")
print(f"Large scale datasets: {datasets_by_scale['Large']}")

# 预处理用于图总结
graph = loader.preprocess_for_summarization(graph)
```

## 安装要求

### 基础要求
- PyTorch
- PyTorch Geometric
- pandas
- numpy

### OGB数据集支持
```bash
pip install ogb
```

注意：如果没有安装OGB包，系统会自动跳过OGB数据集，只支持传统数据集。

## 数据格式

所有数据集统一使用PyTorch Geometric的Data格式：
- `x`: 节点特征矩阵 (num_nodes, feature_dim)
- `edge_index`: 边索引矩阵 (2, num_edges) COO格式
- `y`: 节点标签 (num_nodes,)
- `train_mask`, `val_mask`, `test_mask`: 布尔掩码用于划分训练/验证/测试集