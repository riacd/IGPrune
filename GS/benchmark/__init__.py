"""
基准测试模块

提供标准化的基准测试框架，用于评估不同图总结模型的性能。

功能：
1. 标准化的测试流程
2. 多数据集支持
3. 多种下游任务支持
4. 结果可视化和保存
"""

from .core import Benchmark
from .unified import UnifiedBenchmark

__all__ = ['Benchmark', 'UnifiedBenchmark']