"""
模型注册机制

提供统一的模型注册和管理功能，支持开发模型和baseline模型的统一管理。
"""

from typing import Dict, Type, List, Any
from .base import GraphSummarizationModel
import importlib


class ModelRegistry:
    """
    统一的模型注册表，管理所有Graph Summarization模型。
    """
    
    def __init__(self):
        self._models: Dict[str, Type[GraphSummarizationModel]] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_models()
    
    def register_model(self, 
                      name: str, 
                      model_class: Type[GraphSummarizationModel],
                      category: str = "custom",
                      description: str = "",
                      paper_url: str = "",
                      **kwargs) -> None:
        """
        注册Graph Summarization模型
        
        Args:
            name: 模型名称（唯一标识）
            model_class: 模型类（必须继承GraphSummarizationModel）
            category: 模型分类 ("development", "baseline", "custom")
            description: 模型描述
            paper_url: 相关论文链接
            **kwargs: 其他元信息
        """
        if not issubclass(model_class, GraphSummarizationModel):
            raise ValueError(f"Model {name} must inherit from GraphSummarizationModel")
        
        if name in self._models:
            raise ValueError(f"Model {name} already registered")
        
        self._models[name] = model_class
        self._model_info[name] = {
            "class": model_class,
            "category": category,
            "description": description,
            "paper_url": paper_url,
            **kwargs
        }
    
    def get_model_class(self, name: str) -> Type[GraphSummarizationModel]:
        """获取模型类"""
        if name not in self._models:
            raise ValueError(f"Model {name} not found. Available: {list(self._models.keys())}")
        return self._models[name]
    
    def create_model(self, name: str, **kwargs) -> GraphSummarizationModel:
        """创建模型实例"""
        model_class = self.get_model_class(name)
        return model_class(**kwargs)
    
    def list_models(self, category: str = None) -> List[str]:
        """列出所有模型名称"""
        if category is None:
            return list(self._models.keys())
        return [name for name, info in self._model_info.items() 
                if info.get("category") == category]
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """获取模型信息"""
        if name not in self._model_info:
            raise ValueError(f"Model {name} not found")
        return self._model_info[name].copy()
    
    def list_development_models(self) -> List[str]:
        """列出开发模型"""
        return self.list_models("development")
    
    def list_baseline_models(self) -> List[str]:
        """列出基准模型"""
        return self.list_models("baseline")
    
    def _register_builtin_models(self):
        """注册内置模型"""
        # 旧的learnable模型已被neural_enhanced_gradient系列替代
        # 如需使用旧模型，请直接从main_model导入
        
        try:
            # 注册基于梯度的模型
            from .gradient_based import GradientBasedGraphSummarization

            self.register_model(
                "gradient_based",
                GradientBasedGraphSummarization,
                category="development",
                description="基于梯度的图简化模型（开发模型2）"
            )

        except ImportError as e:
            print(f"Warning: Could not register gradient-based model: {e}")

        # Neural-Enhanced模型通过register_main_models.py注册
        # 避免重复注册


# 全局模型注册表实例
model_registry = ModelRegistry()


def register_model(name: str, model_class: Type[GraphSummarizationModel], **kwargs):
    """便捷函数：注册模型到全局注册表"""
    model_registry.register_model(name, model_class, **kwargs)


def get_model_class(name: str) -> Type[GraphSummarizationModel]:
    """便捷函数：获取模型类"""
    return model_registry.get_model_class(name)


def create_model(name: str, **kwargs) -> GraphSummarizationModel:
    """便捷函数：创建模型实例"""
    return model_registry.create_model(name, **kwargs)


def list_all_models() -> List[str]:
    """便捷函数：列出所有模型"""
    return model_registry.list_models()