# 基准测试模块 (Benchmark Module)

## 模块功能

基准测试模块提供标准化的测试框架，用于评估和比较不同图总结模型的性能。

## 提供的接口

### Benchmark类

主要的基准测试类，提供以下功能：

#### 核心方法
- `run_single_experiment(model, dataset_name, downstream_model)`: 运行单个实验
- `run_comprehensive_benchmark(models, datasets, downstream_models)`: 运行综合基准测试
- `compare_models(models, dataset_name)`: 比较多个模型的性能

#### 实验配置
- `set_experiment_config(num_steps, epochs, device)`: 设置实验参数
- `enable_visualization(save_plots, plot_format)`: 启用结果可视化
- `set_output_directory(results_dir)`: 设置结果保存目录

#### 结果分析
- `generate_performance_report(results)`: 生成性能报告
- `plot_snr_curves(models, complexity_metrics, information_metrics)`: 绘制SNR曲线
- `calculate_statistical_significance(results)`: 计算统计显著性

## 实验流程

1. **数据准备**: 加载和预处理数据集
2. **模型训练**: 使用训练集训练图总结模型
3. **图总结生成**: 生成不同稀疏度的总结图
4. **下游任务评估**: 在总结图上训练和测试下游任务模型
5. **指标计算**: 计算复杂度和信息度量指标
6. **结果保存**: 保存实验结果和可视化图表

## 支持的评估指标

- **Complexity Metric**: 图复杂度（边数的L0范数）
- **Information Metric**: 信息保留度量（基于下游任务损失）
- **SNR-AUC**: Signal-to-Noise Ratio曲线下面积

## 与其它模块的对接

- **datasets模块**: 获取标准化的数据集
- **models模块**: 评估图总结模型和下游任务模型
- **metrics模块**: 使用各种性能度量指标
- **baselines模块**: 集成第三方基准模型

## 输出格式

### 结果文件
- `benchmark_results.csv`: 详细的实验结果数据
- `model_comparison.tsv`: 模型比较摘要
- `snr_curves.png`: SNR曲线可视化

### 数据格式
```csv
model,dataset,step,complexity,information,snr_auc
LearnableGS,Cora,0,10556,0.15,850.2
LearnableGS,Cora,1,8445,0.12,850.2
```

## 使用示例

```python
from GS.benchmark import Benchmark
from GS.models import LearnableGraphSummarization, GCNDownstreamModel

# 初始化基准测试
benchmark = Benchmark(
    results_dir='./results',
    num_steps=10,
    epochs=100
)

# 准备模型
summarizer = LearnableGraphSummarization(input_dim=1433)
downstream = GCNDownstreamModel(input_dim=1433)

# 运行基准测试
results = benchmark.run_single_experiment(
    model=summarizer,
    dataset_name='Cora', 
    downstream_model=downstream
)

# 生成报告
benchmark.generate_performance_report(results)
```

## 配置选项

- `num_steps`: 图总结步数 (默认: 10)
- `epochs`: 下游任务训练轮数 (默认: 100)
- `device`: 计算设备 ('cpu' 或 'cuda')
- `save_intermediate`: 是否保存中间结果 (默认: False)
- `enable_early_stopping`: 是否启用早停 (默认: True)