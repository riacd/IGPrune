# 模型模块 (Models Module)

## 模块功能

模型模块提供图总结任务的核心模型实现，包括：
1. 抽象基类定义
2. 图总结模型实现
3. 下游任务模型实现

## 提供的接口

### 抽象基类

#### GraphSummarizationModel
图总结模型的抽象基类，定义统一接口：
- `summarize(original_graph, num_steps)`: 生成图总结序列
- `reset()`: 重置模型状态

#### DownstreamModel  
下游任务模型的抽象基类：
- `train_model(graph, train_mask, val_mask, labels, epochs)`: 训练模型
- `evaluate(graph, test_mask, labels)`: 评估模型
- `reset()`: 重置模型参数

### 图总结模型

#### LearnableGraphSummarization
本项目开发的主要可学习图总结模型：
- 基于GIN编码器的节点表示学习
- 边重要性评分和渐进式边删除
- 支持多种架构变体和消融实验

#### RandomGraphSummarizationModel
随机边删除基准模型：
- 每步随机删除固定比例的边
- 用于基准比较

#### DummyGraphSummarizationModel
虚拟模型，返回原始图的副本（用于测试）

### 下游任务模型

#### GCNDownstreamModel
基于图卷积网络的节点分类模型：
- 2层GCN架构
- 支持多类别节点分类
- Adam优化器，早停机制

#### GATDownstreamModel  
基于图注意力网络的节点分类模型：
- 多头注意力机制
- Dropout正则化

## 与其它模块的对接

- **datasets模块**: 接收标准化的图数据格式
- **benchmark模块**: 提供标准化的模型评估接口
- **metrics模块**: 配合计算模型性能指标
- **baselines模块**: 统一的图总结模型接口

## 架构设计原则

1. **统一接口**: 所有模型遵循相同的抽象基类
2. **模块化设计**: 图总结和下游任务分离
3. **可扩展性**: 易于添加新的模型变体
4. **类型安全**: 使用类型提示确保参数正确性

## 使用示例

```python
from GS.models import LearnableGraphSummarization, GCNDownstreamModel

# 图总结模型
summarizer = LearnableGraphSummarization(
    input_dim=1433, 
    hidden_dim=256,
    num_steps=10
)

# 生成总结图序列
summary_graphs = summarizer.summarize(original_graph, num_steps=10)

# 下游任务模型
downstream = GCNDownstreamModel(input_dim=1433)
downstream.train_model(graph, train_mask, val_mask, labels)
loss = downstream.evaluate(graph, test_mask, labels)
```