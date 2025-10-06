"""
GS: Graph Summarization Library

这是一个用于图总结任务的综合性Python包，提供：
1. 数据集加载与处理
2. 图总结模型实现  
3. 下游任务模型
4. 基准测试框架
5. 性能度量指标

模块结构：
- datasets: 数据集处理模块
- models: 模型实现模块
- benchmark: 基准测试模块
- metrics: 度量指标模块
- utils: 工具函数模块
"""

__version__ = "1.0.0"

# 导入主要组件
from .datasets import DatasetLoader
from .models import (
    GraphSummarizationModel,
    DownstreamModel,
    NeuralEnhancedGradientModel,
    GradientBasedGraphSummarization,
    GCNDownstreamModel,
    GATDownstreamModel
)
from .benchmark import Benchmark
from .metrics import ComplexityMetric, InformationMetric, SNRAnalysis, ICAnalysis

__all__ = [
    # 数据集
    'DatasetLoader',

    # 模型基类
    'GraphSummarizationModel',
    'DownstreamModel',

    # 图总结模型
    'NeuralEnhancedGradientModel',
    'GradientBasedGraphSummarization',

    # 下游任务模型
    'GCNDownstreamModel',
    'GATDownstreamModel',

    # 基准测试
    'Benchmark',

    # 度量指标
    'ComplexityMetric',
    'InformationMetric',
    'SNRAnalysis'
]