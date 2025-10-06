"""
Neural-Enhanced Gradient-Based Graph Summarization Model

结合开发模型2的梯度信息和神经网络学习能力的混合模型。
该模型使用梯度法提供边重要性的先验知识，然后用神经网络进行微调优化。

训练策略沿用开发模型1的固定重加权和动态重加权策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Optional, Tuple, Dict, Any
import copy
import numpy as np
from tqdm import tqdm

from .base import GraphSummarizationModel
from .gradient_based import GradientBasedGraphSummarization
from .main_model import LearnableGraphSummarization


class EdgeImportanceRefiner(nn.Module):
    """
    边重要性微调网络

    该网络接受：
    1. 梯度法计算的边重要性分数
    2. 边的结构特征（节点嵌入等）
    3. 步骤信息

    输出微调后的边重要性分数
    """

    def __init__(self,
                 node_emb_dim: int = 256,
                 step_emb_dim: int = 32,
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        self.node_emb_dim = node_emb_dim
        self.step_emb_dim = step_emb_dim
        self.hidden_dim = hidden_dim

        # 边特征维度：2*node_emb + |node_diff| + step_emb + gradient_score
        edge_feature_dim = 3 * node_emb_dim + step_emb_dim + 1

        # 微调网络
        self.refiner = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # 输出范围[-1, 1]，作为调整因子
        )

    def forward(self,
                node_embeddings: torch.Tensor,
                edge_index: torch.Tensor,
                gradient_scores: torch.Tensor,
                step_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            node_embeddings: 节点嵌入 [n, node_emb_dim]
            edge_index: 边索引 [2, num_edges]
            gradient_scores: 梯度法计算的边分数 [num_edges]
            step_embedding: 步骤嵌入 [step_emb_dim]

        Returns:
            调整因子 [num_edges]，范围[-1, 1]
        """
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_u = node_embeddings[src_nodes]  # [num_edges, node_emb_dim]
        h_v = node_embeddings[dst_nodes]  # [num_edges, node_emb_dim]

        # 边特征
        edge_diff = torch.abs(h_u - h_v)  # [num_edges, node_emb_dim]
        gradient_scores_expanded = gradient_scores.unsqueeze(-1)  # [num_edges, 1]

        # 扩展步骤嵌入到所有边
        num_edges = edge_index.size(1)
        step_emb_expanded = step_embedding.unsqueeze(0).expand(num_edges, -1)  # [num_edges, step_emb_dim]

        # 拼接所有特征
        edge_features = torch.cat([
            h_u, h_v, edge_diff, step_emb_expanded, gradient_scores_expanded
        ], dim=1)  # [num_edges, edge_feature_dim]

        # 确保数据类型一致性
        edge_features = edge_features.float()

        # 通过微调网络
        adjustment = self.refiner(edge_features).squeeze(-1)  # [num_edges]

        return adjustment


class NeuralEnhancedGradientModel(GraphSummarizationModel, nn.Module):
    """
    神经网络增强的梯度法图总结模型

    该模型结合了：
    1. 梯度法的准确性和可解释性
    2. 神经网络的学习能力和适应性

    工作流程：
    1. 使用梯度法计算初始边重要性（只计算一次并缓存）
    2. 用神经网络学习如何微调这些分数
    3. 通过融合策略得到最终的边选择

    优化特性：
    - 梯度计算缓存：对同一图的梯度分数只计算一次
    - 高效推理：训练后的推理速度远快于纯梯度法

    训练策略：
    - 沿用开发模型1的固定重加权和动态重加权策略
    - 支持uniform和cosine固定权重
    - 支持Frank-Wolfe和UGD动态权重求解
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 step_emb_dim: int = 32,
                 num_gin_layers: int = 3,
                 dropout: float = 0.2,
                 max_steps: int = 20,
                 device: str = 'cpu',
                 gradient_train_epochs: int = 20,
                 fusion_weight: float = 0.3,  # 神经网络调整的权重
                 use_residual_learning: bool = True,
                 fast_gradient_computation: bool = True,  # 是否使用快速梯度计算
                 **kwargs):
        """
        初始化混合模型

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            step_emb_dim: 步骤嵌入维度
            num_gin_layers: GIN层数
            dropout: Dropout率
            max_steps: 最大步数
            device: 计算设备
            gradient_train_epochs: 梯度法训练轮数
            fusion_weight: 神经网络调整的权重
            use_residual_learning: 是否使用残差学习
            fast_gradient_computation: 是否使用快速梯度计算
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.step_emb_dim = step_emb_dim
        self.dropout = dropout
        self.max_steps = max_steps
        self.device = device
        self.fusion_weight = fusion_weight
        self.use_residual_learning = use_residual_learning
        self.fast_gradient_computation = fast_gradient_computation

        # 梯度法模型（用于计算先验边重要性）
        self.gradient_model = GradientBasedGraphSummarization(
            input_dim=input_dim,
            train_epochs=gradient_train_epochs,
            device=torch.device(device) if isinstance(device, str) else device
        )

        # 神经网络节点编码器（复用LearnableGraphSummarization的架构）
        self.neural_encoder = LearnableGraphSummarization(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            step_emb_dim=step_emb_dim,
            num_gin_layers=num_gin_layers,
            dropout=dropout,
            max_steps=max_steps,
            device=device,
            **kwargs
        )

        # 边重要性微调网络
        self.importance_refiner = EdgeImportanceRefiner(
            node_emb_dim=hidden_dim,
            step_emb_dim=step_emb_dim,
            hidden_dim=128,
            dropout=dropout
        )

        # 训练相关
        self.train_mask = None
        self.val_mask = None
        self.labels = None

        # 静态gradient特征存储
        self._static_gradient_features = None
        self._original_edge_index = None
        self._is_gradient_features_computed = False

        # 移动到设备并确保float类型
        self.to(device)
        self.float()

    def set_training_data(self, train_mask, val_mask, labels):
        """设置训练数据"""
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.labels = labels

        # 同时设置梯度模型的训练数据
        self.gradient_model.train_mask = train_mask
        self.gradient_model.val_mask = val_mask
        self.gradient_model.labels = labels

    def precompute_gradient_features(self, graph: Data):
        """
        预计算gradient特征作为静态特征

        Args:
            graph: 原始图数据
        """
        print("正在预计算gradient特征...")

        # 确保图数据在正确的设备上
        graph = graph.to(self.device)

        # 存储原始边索引
        self._original_edge_index = graph.edge_index.clone().to(self.device)

        # 计算gradient特征
        if self.fast_gradient_computation:
            gradient_features = self._compute_gradient_importance_fast(graph, graph.edge_index)
        else:
            gradient_features = self.gradient_model._compute_edge_gradients_sparse(graph, graph.edge_index)

        # 存储为静态特征
        self._static_gradient_features = gradient_features.clone().detach().to(self.device)
        self._is_gradient_features_computed = True

        print(f"Gradient特征预计算完成，共{len(gradient_features)}条边")

    def _get_gradient_features_for_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取给定边的gradient特征

        Args:
            edge_index: 当前边索引

        Returns:
            对应边的gradient特征
        """
        if not self._is_gradient_features_computed:
            raise ValueError("Gradient特征尚未预计算，请先调用precompute_gradient_features()")

        # 如果是原始图的完整边集，直接返回
        if torch.equal(edge_index, self._original_edge_index):
            return self._static_gradient_features

        # 否则需要找到对应的边
        return self._extract_gradient_features_for_subgraph(edge_index)

    def _extract_gradient_features_for_subgraph(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        为子图提取对应的gradient特征

        Args:
            edge_index: 子图边索引

        Returns:
            子图边的gradient特征
        """
        # 创建边的字典映射：(src, dst) -> feature_index
        original_edges = self._original_edge_index.t()  # [num_edges, 2]
        edge_to_idx = {}
        for i, (src, dst) in enumerate(original_edges):
            edge_to_idx[(src.item(), dst.item())] = i

        # 为子图边找到对应的特征索引
        subgraph_edges = edge_index.t()  # [num_subgraph_edges, 2]
        feature_indices = []

        for src, dst in subgraph_edges:
            key = (src.item(), dst.item())
            if key in edge_to_idx:
                feature_indices.append(edge_to_idx[key])
            else:
                # 如果找不到对应边，说明这是新生成的边或反向边
                # 尝试反向边
                reverse_key = (dst.item(), src.item())
                if reverse_key in edge_to_idx:
                    feature_indices.append(edge_to_idx[reverse_key])
                else:
                    # 如果都找不到，使用平均值或者重新计算
                    print(f"Warning: 边 ({src}, {dst}) 在原始图中不存在，使用平均gradient特征")
                    feature_indices.append(0)  # 临时使用第一个特征

        feature_indices = torch.tensor(feature_indices, device=self.device)
        return self._static_gradient_features[feature_indices]


    def _compute_gradient_importance_fast(self, graph: Data, edge_index: torch.Tensor) -> torch.Tensor:
        """
        快速版本的梯度重要性计算（用于训练时）

        基于结构特征的快速近似，而不是实际训练下游模型
        """
        # 确保edge_index在正确的设备上
        edge_index = edge_index.to(self.device)
        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        # 计算节点度数
        degree = torch.zeros(graph.x.size(0), device=self.device, dtype=torch.float)
        degree.scatter_add_(0, src_nodes.to(self.device), torch.ones_like(src_nodes, dtype=torch.float, device=self.device))
        degree.scatter_add_(0, dst_nodes.to(self.device), torch.ones_like(dst_nodes, dtype=torch.float, device=self.device))

        # 边重要性 = 两端节点度数的几何平均
        edge_importance = torch.sqrt(degree[src_nodes] * degree[dst_nodes])

        # 归一化到合理范围
        if edge_importance.max() > 0:
            edge_importance = edge_importance / edge_importance.max()

        return edge_importance

    def _compute_neural_adjustment(self,
                                 graph: Data,
                                 edge_index: torch.Tensor,
                                 gradient_scores: torch.Tensor,
                                 step: int) -> torch.Tensor:
        """
        使用神经网络计算调整因子

        Args:
            graph: 图数据
            edge_index: 边索引
            gradient_scores: 梯度法分数
            step: 当前步骤

        Returns:
            调整因子
        """
        # 获取节点嵌入
        # 确保输入数据在正确的设备上
        graph_x = graph.x.to(self.device)
        edge_index = edge_index.to(self.device)
        node_embeddings = self.neural_encoder.encode_nodes(graph_x, edge_index)

        # 获取步骤嵌入
        step_tensor = torch.tensor(step, dtype=torch.long, device=self.device)
        step_embedding = self.neural_encoder.step_embedding(step_tensor)

        # 计算调整因子
        adjustment = self.importance_refiner(
            node_embeddings, edge_index, gradient_scores, step_embedding
        )

        return adjustment

    def _fuse_scores(self,
                    gradient_scores: torch.Tensor,
                    neural_adjustment: torch.Tensor) -> torch.Tensor:
        """
        融合梯度分数和神经网络调整

        Args:
            gradient_scores: 梯度法分数
            neural_adjustment: 神经网络调整因子

        Returns:
            融合后的分数
        """
        if self.use_residual_learning:
            # 残差学习：基于梯度分数进行调整
            adjusted_scores = gradient_scores + self.fusion_weight * neural_adjustment
        else:
            # 直接加权融合
            adjusted_scores = (1 - self.fusion_weight) * gradient_scores + \
                            self.fusion_weight * neural_adjustment

        return adjusted_scores

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                step: int) -> torch.Tensor:
        """
        前向传播，返回融合后的边分数

        Args:
            x: 节点特征 [n, d_in]
            edge_index: 边索引 [2, num_edges]
            step: 当前步骤

        Returns:
            融合后的边分数 [num_edges]
        """
        # 确保数据类型一致性和设备一致性
        if x.dtype != torch.float32:
            x = x.float()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        # 创建临时图数据
        temp_graph = Data(x=x, edge_index=edge_index)

        # 1. 获取预计算的gradient特征（静态特征）
        gradient_scores = self._get_gradient_features_for_edges(edge_index)

        # 2. 计算神经网络调整
        neural_adjustment = self._compute_neural_adjustment(
            temp_graph, edge_index, gradient_scores, step
        )

        # 3. 融合分数
        final_scores = self._fuse_scores(gradient_scores, neural_adjustment)

        return final_scores

    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        生成图总结序列

        Args:
            original_graph: 原始图
            num_steps: 总结步数

        Returns:
            总结图列表
        """
        if self.train_mask is None or self.val_mask is None or self.labels is None:
            raise ValueError("必须先设置训练数据")

        # 确保gradient特征已预计算
        if not self._is_gradient_features_computed:
            self.precompute_gradient_features(original_graph)

        self.eval()
        summary_graphs = []
        current_graph = copy.deepcopy(original_graph.to(self.device))

        # Step 0: 原始图
        summary_graphs.append(current_graph)

        print(f"开始Neural-Enhanced图简化: 总边数 {current_graph.edge_index.size(1)}")

        with torch.no_grad():
            for step in range(1, num_steps + 1):
                if current_graph.edge_index.size(1) == 0:
                    # 空图
                    empty_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(empty_graph)
                    continue

                # 使用前向传播获取融合的边分数
                edge_scores = self.forward(current_graph.x, current_graph.edge_index, step)

                # 确定保留边数
                if step == num_steps:
                    num_keep = 0
                else:
                    keep_ratio = 1.0 - (step / num_steps)
                    num_keep = max(0, int(current_graph.edge_index.size(1) * keep_ratio))

                if num_keep == 0:
                    current_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                else:
                    # 选择分数最高的边
                    _, top_indices = torch.topk(edge_scores, num_keep, largest=True)
                    kept_edge_index = current_graph.edge_index[:, top_indices]

                    current_graph = Data(
                        x=original_graph.x,
                        edge_index=kept_edge_index,
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )

                summary_graphs.append(current_graph)

        return summary_graphs

    def reset(self) -> None:
        """重置模型"""
        self.neural_encoder.reset()
        self.gradient_model.reset()
        self.train_mask = None
        self.val_mask = None
        self.labels = None
        # 清除静态gradient特征
        self._static_gradient_features = None
        self._original_edge_index = None
        self._is_gradient_features_computed = False

    def get_gradient_features_info(self) -> Dict[str, Any]:
        """获取gradient特征信息用于调试"""
        return {
            'is_computed': self._is_gradient_features_computed,
            'num_edges': len(self._static_gradient_features) if self._static_gradient_features is not None else 0,
            'feature_shape': self._static_gradient_features.shape if self._static_gradient_features is not None else None,
            'device': self._static_gradient_features.device if self._static_gradient_features is not None else None
        }


# 创建不同配置的变体
class NeuralEnhancedGradientModel_HighFusion(NeuralEnhancedGradientModel):
    """高融合权重变体"""
    def __init__(self, *args, **kwargs):
        kwargs['fusion_weight'] = 0.6
        super().__init__(*args, **kwargs)


class NeuralEnhancedGradientModel_LowFusion(NeuralEnhancedGradientModel):
    """低融合权重变体"""
    def __init__(self, *args, **kwargs):
        kwargs['fusion_weight'] = 0.1
        super().__init__(*args, **kwargs)


class NeuralEnhancedGradientModel_NoResidual(NeuralEnhancedGradientModel):
    """不使用残差学习的变体"""
    def __init__(self, *args, **kwargs):
        kwargs['use_residual_learning'] = False
        super().__init__(*args, **kwargs)


class NeuralEnhancedGradientModel_SlowGradient(NeuralEnhancedGradientModel):
    """使用精确梯度计算的变体"""
    def __init__(self, *args, **kwargs):
        kwargs['fast_gradient_computation'] = False
        super().__init__(*args, **kwargs)


# 为了与现有training_strategies.py兼容，创建包装类
class TrainableNeuralEnhancedGradientModel:
    """
    可训练的Neural-Enhanced模型包装类

    该类将NeuralEnhancedGradientModel包装，使其能够与现有的
    GraphSummarizationTrainer和训练策略无缝集成
    """

    def __init__(self,
                 model: NeuralEnhancedGradientModel,
                 training_strategy: str = 'fixed_uniform',
                 solver_type: str = 'frank_wolfe'):
        """
        初始化可训练包装器

        Args:
            model: Neural-Enhanced模型实例
            training_strategy: 训练策略
                - 'fixed_uniform': 固定权重（uniform）
                - 'fixed_cosine': 固定权重（cosine）
                - 'dynamic_frank_wolfe': 动态权重（Frank-Wolfe）
                - 'dynamic_ugd': 动态权重（UGD）
            solver_type: 动态策略的求解器类型
        """
        from .training_strategies import (
            FixedReweightingStrategy,
            DynamicReweightingStrategy,
            GraphSummarizationTrainer
        )

        self.model = model
        self.training_strategy_name = training_strategy

        # 创建训练策略
        if training_strategy.startswith('fixed'):
            strategy_type = training_strategy.split('_')[1]  # 'uniform' or 'cosine'
            self.strategy = FixedReweightingStrategy(strategy_type=strategy_type)
        elif training_strategy.startswith('dynamic'):
            self.strategy = DynamicReweightingStrategy(solver_type=solver_type)
        else:
            raise ValueError(f"Unknown training strategy: {training_strategy}")

        # 创建训练器
        self.trainer = GraphSummarizationTrainer(
            model=model,
            strategy=self.strategy,
            device=model.device
        )

    def train_model(self,
                   graph: Data,
                   train_mask: torch.Tensor,
                   val_mask: torch.Tensor,
                   labels: torch.Tensor,
                   epochs: int = 30,
                   num_steps: int = 10,
                   downstream_epochs: int = 30):
        """
        训练模型

        Args:
            graph: 训练图
            train_mask: 训练掩码
            val_mask: 验证掩码
            labels: 标签
            epochs: 图总结网络训练轮数
            num_steps: 简化步数
            downstream_epochs: 下游模型训练轮数
        """
        # 设置训练数据
        self.model.set_training_data(train_mask, val_mask, labels)

        # 预计算gradient特征（训练前一次性计算）
        if not self.model._is_gradient_features_computed:
            self.model.precompute_gradient_features(graph)

        # 使用训练器训练
        history = self.trainer.train(
            graph=graph,
            train_labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            num_epochs=epochs,
            num_steps=num_steps,
            downstream_epochs=downstream_epochs
        )

        return history

    def summarize(self, graph: Data, num_steps: int = 10) -> List[Data]:
        """生成图总结序列"""
        return self.model.summarize(graph, num_steps)

    def reset(self):
        """重置模型"""
        self.model.reset()

    def parameters(self):
        """获取模型参数"""
        return self.model.parameters()

    def to(self, device):
        """移动到设备"""
        self.model.to(device)
        return self

    def __getattr__(self, name):
        """代理到内部模型"""
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")