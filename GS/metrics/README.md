# 度量模块 (Metrics Module)

## 模块功能

度量模块提供用于评估图总结模型性能的各种指标，包括复杂度度量、信息保留度量和综合分析工具。

## 提供的接口

### ComplexityMetric类

计算图复杂度指标：

#### 方法
- `compute(graph)`: 计算单个图的复杂度
- `compute_list(graph_list)`: 批量计算多个图的复杂度

#### 复杂度定义
- 使用边数的L0范数作为复杂度度量
- 复杂度 = 图中边的总数

### InformationMetric类

计算信息保留度量：

#### 初始化参数
- `downstream_model`: 下游任务模型实例
- `device`: 计算设备

#### 方法  
- `compute(graph, train_mask, val_mask, test_mask, labels, epochs)`: 计算单个图的信息度量
- `compute_list(graph_list, train_mask, val_mask, test_mask, labels, epochs)`: 批量计算

#### 信息度量定义
基于下游任务性能计算信息保留度：
```
Information(G_k) = -log(Loss(G_k) / Loss(G_empty))
```
其中G_k是第k步的总结图，G_empty是空图

### SNRAnalysis类

信号噪声比分析工具：

#### 静态方法
- `compute_snr_auc(complexity_metrics, information_metrics)`: 计算SNR曲线下面积
- `plot_snr_curve(complexity_metrics, information_metrics, title, save_path)`: 绘制SNR曲线
- `compare_snr_curves(results_dict, save_path)`: 比较多个模型的SNR曲线

## 度量指标说明

### Complexity Metric
- **定义**: 图中边数的L0范数
- **范围**: [0, max_edges]
- **意义**: 较低的复杂度表示更简化的图结构

### Information Metric  
- **定义**: 基于下游任务损失的信息保留度量
- **范围**: (-∞, +∞)
- **意义**: 较高的信息度量表示更好的信息保留

### SNR-AUC
- **定义**: SNR曲线与X轴围成的面积
- **计算**: 使用梯形法则积分
- **意义**: 综合评估复杂度-信息权衡性能

## 与其它模块的对接

- **models模块**: 使用下游任务模型计算信息度量
- **datasets模块**: 获取数据集的掩码和标签信息
- **benchmark模块**: 为基准测试提供标准化的度量计算
- **baselines模块**: 评估第三方基准模型性能

## 计算流程

### 信息度量计算流程
1. 重置下游任务模型参数
2. 在总结图上训练下游模型
3. 在测试集上评估模型损失
4. 计算相对于空图的信息保留度

### SNR分析流程  
1. 收集所有步骤的复杂度和信息度量
2. 绘制复杂度-信息度量散点图
3. 计算曲线下面积作为综合指标

## 使用示例

```python
from GS.metrics import ComplexityMetric, InformationMetric, SNRAnalysis
from GS.models import GCNDownstreamModel

# 复杂度度量
complexity_metric = ComplexityMetric()
complexities = complexity_metric.compute_list(summary_graphs)

# 信息度量
downstream_model = GCNDownstreamModel(input_dim=1433)
info_metric = InformationMetric(downstream_model, device='cuda')
informations = info_metric.compute_list(
    summary_graphs, train_mask, val_mask, test_mask, labels
)

# SNR分析
snr_auc = SNRAnalysis.compute_snr_auc(complexities, informations)
SNRAnalysis.plot_snr_curve(complexities, informations, 
                           title="Model Performance", 
                           save_path="snr_curve.png")
```

## 性能优化

- **批量计算**: 使用`compute_list`方法提高批量计算效率
- **GPU加速**: 支持CUDA设备加速下游任务训练
- **内存管理**: 及时释放中间结果避免内存溢出
- **并行处理**: 支持多进程并行计算不同图的度量