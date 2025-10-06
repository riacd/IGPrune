"""
训练策略变体

将训练策略作为LearnableGraphSummarization模型的消融实验变体。
根据MODEL.md，训练策略是模型的一部分，而不是独立的组件。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Union
import math
import numpy as np
import torch
from torch_geometric.data import Data

from .main_model import LearnableGraphSummarization
from .training_strategies import (
    FixedReweightingStrategy, 
    DynamicReweightingStrategy,
    GraphSummarizationTrainer
)
from .gradient_based import GradientBasedGraphSummarization
from .base import GraphSummarizationModel


class LearnableGraphSummarization_FixedUniform(LearnableGraphSummarization):
    """
    使用固定均匀权重训练策略的模型变体。
    
    训练策略：所有步骤权重均为1（uniform weighting）
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = FixedReweightingStrategy('uniform')
        self.variant_name = "fixed_uniform"
    
    def get_training_strategy(self):
        """返回该变体使用的训练策略"""
        return self.training_strategy


class LearnableGraphSummarization_FixedCosine(LearnableGraphSummarization):
    """
    使用固定余弦权重训练策略的模型变体。
    
    训练策略：权重 = 0.5 + 0.5*cos(k/N_step)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = FixedReweightingStrategy('cosine')
        self.variant_name = "fixed_cosine"
    
    def get_training_strategy(self):
        """返回该变体使用的训练策略"""
        return self.training_strategy


class LearnableGraphSummarization_DynamicFW(LearnableGraphSummarization):
    """
    使用动态Frank-Wolfe权重训练策略的模型变体。
    
    训练策略：使用Frank-Wolfe算法动态计算权重
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = DynamicReweightingStrategy('frank_wolfe')
        self.variant_name = "dynamic_frank_wolfe"
    
    def get_training_strategy(self):
        """返回该变体使用的训练策略"""
        return self.training_strategy


class LearnableGraphSummarization_DynamicUGD(LearnableGraphSummarization):
    """
    使用动态UGD权重训练策略的模型变体。
    
    训练策略：使用UGD算法动态计算权重
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.training_strategy = DynamicReweightingStrategy('ugd')
        self.variant_name = "dynamic_ugd"
    
    def get_training_strategy(self):
        """返回该变体使用的训练策略"""
        return self.training_strategy


class TrainableGraphSummarizationModel:
    """
    可训练的图总结模型包装器。
    
    将模型和其对应的训练策略结合在一起，提供统一的训练接口。
    """
    
    def __init__(self, model: Union[LearnableGraphSummarization, GradientBasedGraphSummarization], downstream_model_factory=None):
        """
        初始化可训练模型。
        
        Args:
            model: 图总结模型
            downstream_model_factory: 创建下游任务模型的工厂函数，如果为None则使用默认GCN
        """
        self.model = model
        self.downstream_model_factory = downstream_model_factory
        
        # 根据模型类型选择不同的训练方式
        if isinstance(model, GradientBasedGraphSummarization):
            # 基于梯度的模型使用特殊训练方式
            self.training_strategy = None
            self.trainer = None
        elif isinstance(model, LearnableGraphSummarization):
            # 可学习模型使用标准训练策略
            if hasattr(model, 'training_strategy'):
                self.training_strategy = model.training_strategy
            else:
                # 默认使用固定均匀权重策略
                self.training_strategy = FixedReweightingStrategy('uniform')
            
            # 创建训练器
            self.trainer = GraphSummarizationTrainer(
                model=self.model,
                strategy=self.training_strategy,
                downstream_model_factory=downstream_model_factory
            )
        else:
            # 其他模型暂不支持训练
            self.training_strategy = None
            self.trainer = None
    
    def train(self, 
              graph: Data,
              train_labels: torch.Tensor,
              train_mask: torch.Tensor,
              val_mask: torch.Tensor,
              *args, **kwargs):
        """训练模型"""
        if isinstance(self.model, GradientBasedGraphSummarization):
            # 基于梯度的模型需要设置训练数据
            self.model.train_mask = train_mask.to(self.model.device)
            self.model.val_mask = val_mask.to(self.model.device)  
            self.model.labels = train_labels.to(self.model.device)
            self.model.is_trained = True
            
            # 返回空的训练历史（基于梯度的模型不需要传统训练）
            return {'loss_history': [], 'val_loss_history': []}
        elif self.trainer is not None:
            # 可学习模型使用标准训练器
            return self.trainer.train(graph, train_labels, train_mask, val_mask, *args, **kwargs)
        else:
            # 不支持训练的模型
            print("Warning: 该模型不支持训练")
            return {'loss_history': [], 'val_loss_history': []}
    
    def summarize(self, *args, **kwargs):
        """生成图总结（委托给底层模型）"""
        return self.model.summarize(*args, **kwargs)
    
    def get_variant_info(self) -> Dict[str, Any]:
        """获取变体信息"""
        return {
            'model_type': type(self.model).__name__,
            'training_strategy': type(self.training_strategy).__name__,
            'variant_name': getattr(self.model, 'variant_name', 'unknown'),
            'node_encoder': getattr(self.model, 'node_encoder_type', 'gin'),
            'hidden_dim': getattr(self.model, 'hidden_dim', 256),
            'use_step_embedding': getattr(self.model, 'use_step_embedding', True),
            'use_edge_diff': getattr(self.model, 'use_edge_diff', True)
        }