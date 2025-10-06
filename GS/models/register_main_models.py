"""
注册Neural-Enhanced模型及其消融实验变体

专注于neural_enhanced_gradient系列模型，已替代旧的learnable模型。
"""

from .registry import model_registry
from .neural_enhanced_gradient import (
    NeuralEnhancedGradientModel,
    NeuralEnhancedGradientModel_HighFusion,
    NeuralEnhancedGradientModel_LowFusion,
    NeuralEnhancedGradientModel_NoResidual,
    NeuralEnhancedGradientModel_SlowGradient
)


def register_all_main_models():
    """注册Neural-Enhanced图总结模型及其变体"""

    # 主要模型 (标准配置)
    model_registry.register_model(
        name="neural_enhanced_main",
        model_class=NeuralEnhancedGradientModel,
        description="Neural-Enhanced gradient-based graph summarization model (default config)",
        category="development",
        variant="base"
    )

    # 融合权重变体 (Ablation Study)
    model_registry.register_model(
        name="neural_enhanced_high_fusion",
        model_class=NeuralEnhancedGradientModel_HighFusion,
        description="Neural-Enhanced model with high fusion weight (0.6)",
        category="development",
        variant="fusion_weight"
    )

    model_registry.register_model(
        name="neural_enhanced_low_fusion",
        model_class=NeuralEnhancedGradientModel_LowFusion,
        description="Neural-Enhanced model with low fusion weight (0.1)",
        category="development",
        variant="fusion_weight"
    )

    # 学习策略变体 (Ablation Study)
    model_registry.register_model(
        name="neural_enhanced_no_residual",
        model_class=NeuralEnhancedGradientModel_NoResidual,
        description="Neural-Enhanced model without residual learning",
        category="development",
        variant="learning_strategy"
    )

    # 梯度计算变体 (Ablation Study)
    model_registry.register_model(
        name="neural_enhanced_slow_gradient",
        model_class=NeuralEnhancedGradientModel_SlowGradient,
        description="Neural-Enhanced model with exact gradient computation",
        category="development",
        variant="gradient_computation"
    )

    print("✅ Registered Neural-Enhanced model variants")


def get_available_training_strategies():
    """获取可用的训练策略列表"""
    return ['fixed_uniform', 'fixed_cosine', 'dynamic_frank_wolfe', 'dynamic_ugd']


if __name__ == "__main__":
    register_all_main_models()