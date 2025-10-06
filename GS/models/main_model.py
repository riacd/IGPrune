"""
主要的图总结模型实现

根据MODEL.md文档的详细规格实现LearnableGraphSummarization模型，
包括完整的神经网络架构和训练策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GATConv, SAGEConv
from typing import List, Optional, Dict, Any, Callable
import copy
import math
import numpy as np
from abc import abstractmethod

from .base import GraphSummarizationModel


class GINLayer(nn.Module):
    """
    GIN层实现，包含BatchNorm、ReLU和Dropout。
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 eps: float = 0.0, 
                 train_eps: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        
        self.gin_conv = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            ),
            eps=eps,
            train_eps=train_eps
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.gin_conv(x, edge_index)
        x = self.dropout(x)
        return x


class EdgeScorer(nn.Module):
    """
    边分类器MLP，输出每条边的移除概率。
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.mlp = nn.Sequential(
            # FC1: input -> 512
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            # FC2: 512 -> 256  
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            # FC3: 256 -> 1
            nn.Linear(256, 1)
        )
        
    def forward(self, edge_features):
        return self.mlp(edge_features)


class LearnableGraphSummarization(GraphSummarizationModel, nn.Module):
    """
    可学习的图总结模型，完整实现MODEL.md中定义的架构。
    
    Architecture:
    - Node Encoder: 3-layer GIN with hidden_dim=256
    - Step Embedding: Learnable lookup table with d_s=32
    - Edge Scorer: 3-layer MLP (input -> 512 -> 256 -> 1)
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 step_emb_dim: int = 32,
                 num_gin_layers: int = 3,
                 dropout: float = 0.2,
                 max_steps: int = 20,
                 device: str = 'cpu',
                 node_encoder_type: str = 'gin',  # 'gin', 'gat', 'sage'
                 use_step_embedding: bool = True,
                 use_edge_diff: bool = True,
                 **kwargs):
        """
        初始化模型。
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度 (默认256)
            step_emb_dim: 步骤嵌入维度 (默认32)
            num_gin_layers: GIN层数 (默认3)
            dropout: Dropout率 (默认0.2)
            max_steps: 最大步数
            device: 计算设备
            node_encoder_type: 节点编码器类型
            use_step_embedding: 是否使用步骤嵌入
            use_edge_diff: 是否使用边差异特征
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.step_emb_dim = step_emb_dim
        self.num_gin_layers = num_gin_layers
        self.dropout = dropout
        self.max_steps = max_steps
        self.device = device
        self.node_encoder_type = node_encoder_type
        self.use_step_embedding = use_step_embedding
        self.use_edge_diff = use_edge_diff
        
        # 初始化节点编码器
        self._build_node_encoder()
        
        # 步骤嵌入
        if self.use_step_embedding:
            self.step_embedding = nn.Embedding(max_steps + 1, step_emb_dim)
        
        # 边表示维度计算
        edge_repr_dim = 2 * hidden_dim  # h_u + h_v
        if self.use_edge_diff:
            edge_repr_dim += hidden_dim  # |h_u - h_v|
        if self.use_step_embedding:
            edge_repr_dim += step_emb_dim  # step embedding
            
        # 边分类器
        self.edge_scorer = EdgeScorer(edge_repr_dim, dropout)
        
        # Xavier初始化
        self._initialize_weights()
        
        # 确保模型参数都是float32类型
        self.float()
    
    def _build_node_encoder(self):
        """构建节点编码器"""
        if self.node_encoder_type == 'gin':
            # 输入投影层
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
            
            # GIN层
            self.gin_layers = nn.ModuleList([
                GINLayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    eps=0.0,
                    train_eps=True,
                    dropout=self.dropout
                ) for _ in range(self.num_gin_layers)
            ])
            
        elif self.node_encoder_type == 'gat':
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
            self.gat_layers = nn.ModuleList([
                GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    dropout=self.dropout
                ) for _ in range(self.num_gin_layers)
            ])
            
        elif self.node_encoder_type == 'sage':
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
            self.sage_layers = nn.ModuleList([
                SAGEConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim
                ) for _ in range(self.num_gin_layers)
            ])
        else:
            raise ValueError(f"Unsupported encoder type: {self.node_encoder_type}")
    
    def _initialize_weights(self):
        """Xavier uniform初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        节点编码。
        
        Args:
            x: 节点特征 [n, d_in]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            节点嵌入 [n, d_h]
        """
        # 确保输入是float32类型
        if x.dtype != torch.float32:
            x = x.float()
        
        # 输入投影
        h = self.input_projection(x)
        
        # 通过编码器层
        if self.node_encoder_type == 'gin':
            for gin_layer in self.gin_layers:
                h = gin_layer(h, edge_index)
        elif self.node_encoder_type == 'gat':
            for gat_layer in self.gat_layers:
                h = F.relu(gat_layer(h, edge_index))
                h = F.dropout(h, p=self.dropout, training=self.training)
        elif self.node_encoder_type == 'sage':
            for sage_layer in self.sage_layers:
                h = F.relu(sage_layer(h, edge_index))
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def compute_edge_features(self, 
                            node_embeddings: torch.Tensor,
                            edge_index: torch.Tensor,
                            step: int) -> torch.Tensor:
        """
        计算边特征表示。
        
        Args:
            node_embeddings: 节点嵌入 [n, d_h]
            edge_index: 边索引 [2, num_edges]
            step: 当前步骤
            
        Returns:
            边特征 [num_edges, edge_feature_dim]
        """
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_u = node_embeddings[src_nodes]  # [num_edges, d_h]
        h_v = node_embeddings[dst_nodes]  # [num_edges, d_h]
        
        # 基本边特征: [h_u; h_v]
        edge_features = [h_u, h_v]
        
        # 差异特征: |h_u - h_v|
        if self.use_edge_diff:
            edge_diff = torch.abs(h_u - h_v)
            edge_features.append(edge_diff)
        
        # 步骤嵌入
        if self.use_step_embedding:
            step_tensor = torch.full((edge_index.size(1),), step, 
                                   dtype=torch.long, device=edge_index.device)
            step_emb = self.step_embedding(step_tensor)  # [num_edges, d_s]
            edge_features.append(step_emb)
        
        # 拼接所有特征
        edge_repr = torch.cat(edge_features, dim=1)
        return edge_repr
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                step: int) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 节点特征 [n, d_in]
            edge_index: 边索引 [2, num_edges]
            step: 当前步骤
            
        Returns:
            边分数 [num_edges]
        """
        # 确保数据类型一致性
        if x.dtype != torch.float32:
            x = x.float()
        
        # 节点编码
        node_embeddings = self.encode_nodes(x, edge_index)
        
        # 边特征计算
        edge_features = self.compute_edge_features(node_embeddings, edge_index, step)
        
        # 边分类
        edge_scores = self.edge_scorer(edge_features).squeeze(-1)
        
        return edge_scores
    
    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        生成图总结序列。
        
        Args:
            original_graph: 原始图
            num_steps: 总结步数
            
        Returns:
            总结图列表，包括原图在内共num_steps+1个图
        """
        self.eval()
        summary_graphs = []
        current_graph = copy.deepcopy(original_graph)
        
        # Step 0: 原始图
        summary_graphs.append(current_graph)
        
        with torch.no_grad():
            for step in range(1, num_steps + 1):
                if current_graph.edge_index.size(1) == 0:
                    # 已经是空图，后续都是空图
                    empty_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long, 
                                             device=original_graph.edge_index.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(empty_graph)
                    continue
                
                # 计算边分数
                edge_scores = self.forward(current_graph.x, current_graph.edge_index, step)
                
                # 确定保留边数
                current_edges = current_graph.edge_index.size(1)
                keep_ratio = 1.0 - (step / num_steps)
                num_keep = max(0, int(current_edges * keep_ratio))
                
                if num_keep == 0:
                    # 空图
                    empty_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long,
                                             device=original_graph.edge_index.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(empty_graph)
                    current_graph = empty_graph
                else:
                    # 选择分数最高的边
                    _, top_indices = torch.topk(edge_scores, num_keep, largest=True)
                    kept_edge_index = current_graph.edge_index[:, top_indices]
                    
                    next_graph = Data(
                        x=original_graph.x,
                        edge_index=kept_edge_index,
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(next_graph)
                    current_graph = next_graph
        
        return summary_graphs
    
    def reset(self) -> None:
        """重置模型参数"""
        self._initialize_weights()


# 消融实验变体
class LearnableGraphSummarization_GAT(LearnableGraphSummarization):
    """使用GAT作为节点编码器的变体"""
    
    def __init__(self, *args, **kwargs):
        kwargs['node_encoder_type'] = 'gat'
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_SAGE(LearnableGraphSummarization):
    """使用GraphSAGE作为节点编码器的变体"""
    
    def __init__(self, *args, **kwargs):
        kwargs['node_encoder_type'] = 'sage'
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_NoStepEmb(LearnableGraphSummarization):
    """不使用步骤嵌入的变体"""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_step_embedding'] = False
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_NoEdgeDiff(LearnableGraphSummarization):
    """不使用边差异特征的变体"""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_edge_diff'] = False
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_SmallHidden(LearnableGraphSummarization):
    """使用较小隐藏维度的变体"""
    
    def __init__(self, input_dim, *args, **kwargs):
        kwargs['hidden_dim'] = 128
        super().__init__(input_dim, *args, **kwargs)


class LearnableGraphSummarization_LargeHidden(LearnableGraphSummarization):
    """使用较大隐藏维度的变体"""
    
    def __init__(self, input_dim, *args, **kwargs):
        kwargs['hidden_dim'] = 512
        super().__init__(input_dim, *args, **kwargs)


class LearnableGraphSummarization_DeepGIN(LearnableGraphSummarization):
    """使用更深GIN网络的变体"""
    
    def __init__(self, *args, **kwargs):
        kwargs['num_gin_layers'] = 5
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_ShallowGIN(LearnableGraphSummarization):
    """使用较浅GIN网络的变体"""
    
    def __init__(self, *args, **kwargs):
        kwargs['num_gin_layers'] = 2
        super().__init__(*args, **kwargs)