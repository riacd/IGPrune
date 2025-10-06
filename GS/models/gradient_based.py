"""
基于梯度的图简化模型

实现开发模型2：使用梯度信息逐步删除边的图简化方法。
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from typing import List, Optional
import copy
import numpy as np
import time
from tqdm import tqdm

from .base import GraphSummarizationModel
from .downstream import GCNDownstreamModel


class GradientBasedGraphSummarization(GraphSummarizationModel):
    """
    基于梯度的图简化模型
    
    该模型通过以下步骤实现图简化：
    1. 在训练集上训练下游任务模型
    2. 在验证集上进行预测，计算损失
    3. 反向传播得到边权重的梯度
    4. 删除梯度最小的边（保留对性能影响最小的边）
    5. 重复上述过程直到所有边被删除
    
    注意：此模型在训练阶段学习如何选择边，但与可学习模型不同，
    它使用梯度信息而不是神经网络参数来决定删除哪些边。
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 downstream_model_type: str = 'gcn',
                 hidden_dim: int = 128,
                 train_epochs: int = 30,
                 device: Optional[torch.device] = None):
        """
        初始化基于梯度的图简化模型
        
        Args:
            input_dim: 输入特征维度（用于benchmark兼容性）
            downstream_model_type: 下游任务模型类型 ('gcn' 或 'gat')
            hidden_dim: 隐藏层维度
            train_epochs: 训练轮数
            device: 计算设备
        """
        self.input_dim = input_dim
        self.downstream_model_type = downstream_model_type
        self.hidden_dim = hidden_dim
        self.train_epochs = train_epochs
        self.device = device if device is not None else torch.device('cpu')
        
        # 存储训练数据的引用（将在训练时设置）
        self.train_mask = None
        self.val_mask = None  
        self.labels = None
        
        # 标记模型是否已训练
        self.is_trained = False
    
    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        生成一系列简化后的图
        
        Args:
            original_graph: 原始输入图
            num_steps: 简化步数
            
        Returns:
            List[Data]: 简化后的图列表，包含原图和逐步简化的结果
        """
        if self.train_mask is None or self.val_mask is None or self.labels is None:
            raise ValueError("必须先设置训练数据。请确保模型已通过TrainableGraphSummarizationModel进行训练。")
        
        # 将图移动到指定设备
        graph = original_graph.to(self.device)
        
        # 结果列表，从原图开始
        summarized_graphs = [copy.deepcopy(graph)]
        
        # 使用稀疏表示以提高效率
        edge_index = graph.edge_index
        current_edges = edge_index.clone()
        
        # 计算每步需要删除的边数
        total_edges = current_edges.shape[1]
        edges_per_step = total_edges // num_steps
        remaining_edges = total_edges
        
        print(f"开始梯度法图简化: 总边数 {total_edges}, 每步删除 {edges_per_step} 边")
        print("这可能需要一些时间，请耐心等待...")
        
        for step in tqdm(range(num_steps), desc="图简化进度"):
            if current_edges.shape[1] <= 0:
                # 如果没有边了，添加空图
                empty_graph = copy.deepcopy(graph)
                empty_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                summarized_graphs.append(empty_graph)
                continue
            
            # 计算当前步骤要删除的边数
            if step == num_steps - 1:
                # 最后一步删除所有剩余边
                edges_to_remove = current_edges.shape[1]
            else:
                edges_to_remove = min(edges_per_step, current_edges.shape[1])
            
            print(f"步骤 {step+1}: 当前边数 {current_edges.shape[1]}, 删除 {edges_to_remove} 边")

            # 获取边的梯度信息（使用稀疏方法）
            print(f"  正在计算 {current_edges.shape[1]} 条边的重要性...")
            start_time = time.time()
            edge_gradients = self._compute_edge_gradients_sparse(graph, current_edges)
            elapsed_time = time.time() - start_time
            print(f"  边重要性计算完成，耗时 {elapsed_time:.1f} 秒")
            
            # 选择要删除的边（梯度最小的边）
            current_edges = self._remove_edges_by_gradient_sparse(current_edges, edge_gradients, edges_to_remove)
            
            # 创建新的图数据
            new_graph = copy.deepcopy(graph)
            new_graph.edge_index = current_edges
            
            summarized_graphs.append(new_graph)
        
        return summarized_graphs
    
    def _compute_edge_gradients_sparse(self, graph: Data, edge_index: torch.Tensor) -> torch.Tensor:
        """
        使用稀疏表示计算边权重的梯度
        
        Args:
            graph: 当前图数据
            edge_index: 当前边索引
            
        Returns:
            torch.Tensor: 每条边的梯度值
        """
        # 创建下游任务模型
        input_dim = graph.x.size(1)
        output_dim = int(self.labels.max()) + 1
        
        if self.downstream_model_type.lower() == 'gcn':
            downstream_model = GCNDownstreamModel(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                device=self.device
            )
        else:
            raise ValueError(f"不支持的下游模型类型: {self.downstream_model_type}")
        
        # 创建临时图数据进行训练
        temp_graph = copy.deepcopy(graph)
        temp_graph.edge_index = edge_index.detach()
        
        # 训练下游模型
        downstream_model.train_model(
            temp_graph, 
            self.train_mask, 
            self.val_mask, 
            self.labels, 
            epochs=self.train_epochs
        )
        
        # 为每条边计算梯度
        num_edges = edge_index.shape[1]
        edge_gradients = torch.zeros(num_edges, device=self.device)
        
        downstream_model.model.eval()
        
        # 批量计算边的重要性（简化版本：通过删除每条边看性能下降）
        with torch.no_grad():
            # 计算完整图的性能作为基准
            base_out = downstream_model.model(graph.x.float(), edge_index)
            base_loss = F.nll_loss(base_out[self.val_mask], self.labels[self.val_mask].to(base_out.device))
            
            # 对每条边计算删除后的性能变化
            batch_size = min(50, num_edges)  # 减小批次大小以节省内存
            print(f"      使用批次大小 {batch_size} 处理 {num_edges} 条边...")

            for i in tqdm(range(0, num_edges, batch_size), desc="      边重要性计算", leave=False):
                end_idx = min(i + batch_size, num_edges)
                batch_gradients = []
                
                for edge_idx in range(i, end_idx):
                    # 创建删除当前边的边索引
                    mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)
                    mask[edge_idx] = False
                    reduced_edge_index = edge_index[:, mask]
                    
                    # 计算删除边后的性能
                    if reduced_edge_index.shape[1] > 0:
                        out = downstream_model.model(graph.x.float(), reduced_edge_index)
                        loss = F.nll_loss(out[self.val_mask], self.labels[self.val_mask].to(out.device))
                        gradient = loss - base_loss  # 损失增加越多，边越重要
                    else:
                        gradient = float('inf')  # 如果删除这条边导致图断开，设为无穷大
                    
                    batch_gradients.append(gradient)
                
                edge_gradients[i:end_idx] = torch.tensor(batch_gradients, device=self.device)
        
        return edge_gradients
    
    def _remove_edges_by_gradient_sparse(self, edge_index: torch.Tensor, gradients: torch.Tensor, num_edges_to_remove: int) -> torch.Tensor:
        """
        根据梯度信息删除边（稀疏版本）
        
        Args:
            edge_index: 当前边索引
            gradients: 边的梯度信息
            num_edges_to_remove: 要删除的边数
            
        Returns:
            torch.Tensor: 删除边后的边索引
        """
        if num_edges_to_remove >= edge_index.shape[1]:
            # 删除所有边
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # 找到梯度最小的边（删除对性能影响最小的边）
        _, indices_to_remove = torch.topk(gradients, k=num_edges_to_remove, largest=False)
        
        # 创建保留边的掩码
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=self.device)
        mask[indices_to_remove] = False
        
        # 返回保留的边
        return edge_index[:, mask]
    
    def _compute_edge_gradients(self, graph: Data, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算边权重的梯度
        
        Args:
            graph: 当前图数据
            adj_matrix: 当前邻接矩阵（需要梯度）
            
        Returns:
            torch.Tensor: 边的梯度信息
        """
        # 创建下游任务模型
        input_dim = graph.x.size(1)
        output_dim = int(self.labels.max()) + 1
        
        if self.downstream_model_type.lower() == 'gcn':
            downstream_model = GCNDownstreamModel(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                device=self.device
            )
        else:
            raise ValueError(f"不支持的下游模型类型: {self.downstream_model_type}")
        
        # 使用当前邻接矩阵创建边索引
        edge_index, _ = dense_to_sparse(adj_matrix.detach())
        temp_graph = copy.deepcopy(graph)
        temp_graph.edge_index = edge_index
        
        # 训练下游模型
        downstream_model.train_model(
            temp_graph, 
            self.train_mask, 
            self.val_mask, 
            self.labels, 
            epochs=self.train_epochs
        )
        
        # 在验证集上计算损失并求梯度
        downstream_model.model.eval()
        
        # 重新创建需要梯度的邻接矩阵
        adj_for_grad = adj_matrix.clone().detach().requires_grad_(True)
        
        # 前向传播
        edge_index_grad, _ = dense_to_sparse(adj_for_grad)
        out = downstream_model.model(graph.x.float(), edge_index_grad)
        
        # 计算验证损失
        val_loss = F.nll_loss(out[self.val_mask], self.labels[self.val_mask].to(out.device))
        
        # 反向传播
        val_loss.backward()
        
        # 检查梯度是否存在
        if adj_for_grad.grad is None:
            print("警告: 邻接矩阵梯度为None，返回零梯度")
            return torch.zeros_like(adj_for_grad)
        
        # 返回梯度
        return adj_for_grad.grad.clone()
    
    def _remove_edges_by_gradient(self, adj_matrix: torch.Tensor, gradients: torch.Tensor, num_edges_to_remove: int) -> torch.Tensor:
        """
        根据梯度信息删除边
        
        Args:
            adj_matrix: 当前邻接矩阵
            gradients: 边的梯度信息
            num_edges_to_remove: 要删除的边数
            
        Returns:
            torch.Tensor: 删除边后的邻接矩阵
        """
        # 只考虑上三角部分（无向图）
        mask = torch.triu(adj_matrix.bool(), diagonal=1)
        
        # 获取现有边的位置和对应的梯度
        edge_positions = mask.nonzero(as_tuple=False)
        
        if len(edge_positions) == 0:
            return adj_matrix
        
        # 获取这些边对应的梯度值
        edge_grads = gradients[edge_positions[:, 0], edge_positions[:, 1]]
        
        # 找到梯度最小的边（删除对性能影响最小的边）
        _, indices_to_remove = torch.topk(edge_grads, k=min(num_edges_to_remove, len(edge_grads)), largest=False)
        
        # 创建新的邻接矩阵
        new_adj = adj_matrix.clone().detach()
        
        for idx in indices_to_remove:
            i, j = edge_positions[idx]
            # 删除无向边（对称删除）
            new_adj[i, j] = 0
            new_adj[j, i] = 0
        
        return new_adj
    
    def reset(self) -> None:
        """重置模型状态"""
        self.train_mask = None
        self.val_mask = None
        self.labels = None