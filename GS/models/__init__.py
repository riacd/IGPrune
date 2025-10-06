"""
模型模块

包含Neural-Enhanced图总结模型和相关组件的实现。

模块结构：
- base: 抽象基类定义
- neural_enhanced_gradient: Neural-Enhanced图总结模型及其变体
- gradient_based: 基础梯度法模型
- downstream: 下游任务模型实现
- registry: 统一的模型注册机制

消融实验变体分类：
1. 融合权重变体（高融合、低融合等）
2. 学习策略变体（残差学习、直接融合等）
3. 计算精度变体（快速计算、精确计算等）
"""

# 基类
from .base import GraphSummarizationModel, DownstreamModel

# Neural-Enhanced图总结模型及其变体
from .neural_enhanced_gradient import (
    NeuralEnhancedGradientModel,
    NeuralEnhancedGradientModel_HighFusion,
    NeuralEnhancedGradientModel_LowFusion,
    NeuralEnhancedGradientModel_NoResidual,
    NeuralEnhancedGradientModel_SlowGradient,
    TrainableNeuralEnhancedGradientModel,
    EdgeImportanceRefiner
)

# 基础梯度法模型
from .gradient_based import GradientBasedGraphSummarization

# 下游任务模型
from .downstream import (
    GCNDownstreamModel,
    GATDownstreamModel,
    GCNModel,
    GATModel
)

# 模型注册机制
from .registry import (
    model_registry,
    register_model,
    get_model_class,
    create_model,
    list_all_models
)

# 自动注册所有模型
try:
    # 注册主要模型变体
    from .register_main_models import register_all_main_models
    register_all_main_models()
except ImportError:
    pass  # 注册失败时静默失败

# 注册baseline模型
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from baselines.register_baselines import register_all_baselines
    register_all_baselines()
except ImportError:
    pass  # baseline模块不可用时静默失败

__all__ = [
    # 基类
    'GraphSummarizationModel',
    'DownstreamModel',

    # Neural-Enhanced图总结模型及其变体
    'NeuralEnhancedGradientModel',
    'NeuralEnhancedGradientModel_HighFusion',
    'NeuralEnhancedGradientModel_LowFusion',
    'NeuralEnhancedGradientModel_NoResidual',
    'NeuralEnhancedGradientModel_SlowGradient',
    'TrainableNeuralEnhancedGradientModel',
    'EdgeImportanceRefiner',

    # 基础梯度法模型
    'GradientBasedGraphSummarization',

    # 下游任务模型
    'GCNDownstreamModel',
    'GATDownstreamModel',
    'GCNModel',
    'GATModel',

    # 模型注册机制
    'model_registry',
    'register_model',
    'get_model_class',
    'create_model',
    'list_all_models'
]