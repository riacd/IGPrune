"""
度量模块

提供用于评估图总结模型性能的各种度量指标。

包含：
- ComplexityMetric: 计算图复杂度（边数）
- InformationMetric: 计算信息保留度量（支持双重归一化）
- AccuracyMetric: 计算下游任务准确度
- ICAnalysis: 计算IC-AUC和信息阈值点等分析指标
- SNRAnalysis: 旧版名称，向后兼容
"""

from .core import ComplexityMetric, InformationMetric, AccuracyMetric, ICAnalysis, SNRAnalysis

__all__ = ['ComplexityMetric', 'InformationMetric', 'AccuracyMetric', 'ICAnalysis', 'SNRAnalysis']