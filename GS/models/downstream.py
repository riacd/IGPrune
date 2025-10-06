"""
下游任务模型实现

包含GCN、GAT等用于节点分类等下游任务的模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.models import GAE, VGAE
from torch_geometric.utils import negative_sampling
from typing import Optional
from tqdm import tqdm
from .base import DownstreamModel


class GCNDownstreamModel(DownstreamModel):
    """
    GCN model for downstream tasks like node classification.
    
    This model uses a 2-layer GCN architecture and can be trained
    on any graph structure.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            device: Computation device
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device if device is not None else torch.device('cpu')
        
        self.model = None
        self.optimizer = None
        self._build_model()
        
    def _build_model(self):
        """Build the GCN model."""
        if self.output_dim is None:
            # Will be set during first training call
            return
            
        self.model = GCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim, 
            output_dim=self.output_dim
        ).to(self.device).float()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )
    
    def train_model(self,
                    graph: Data,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    labels: torch.Tensor,
                    epochs: int = 100) -> None:
        """Train the GCN model with memory optimization."""
        # Set output dimension if not set
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        self.model.train()

        # Ensure float32 dtype for compatibility and move to device
        if graph.x.dtype == torch.float64:
            graph.x = graph.x.float()
        graph = graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        labels = labels.to(self.device)

        # Memory-optimized training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            try:
                # Forward pass
                out = self.model(graph.x, graph.edge_index)
                loss = F.nll_loss(out[train_mask], labels[train_mask])

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Clear intermediate gradients to save memory
                if epoch % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Optional: early stopping based on validation loss
                if epoch % 50 == 0 and val_mask.sum() > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(graph.x, graph.edge_index)
                        val_loss = F.nll_loss(val_out[val_mask], labels[val_mask])
                    self.model.train()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ Memory error at epoch {epoch}, clearing cache and continuing...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Retry with gradient checkpointing or skip this epoch
                    continue
                else:
                    raise e
    
    def evaluate(self,
                 graph: Data,
                 test_mask: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """Evaluate model on test set."""
        self.model.eval()

        # Ensure float32 dtype for compatibility
        if graph.x.dtype == torch.float64:
            graph.x = graph.x.float()

        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            test_loss = F.nll_loss(out[test_mask], labels[test_mask].to(out.device))

        return float(test_loss)

    def predict(self, graph: Data) -> torch.Tensor:
        """Generate predictions for all nodes in the graph."""
        self.model.eval()

        # Ensure float32 dtype for compatibility
        if graph.x.dtype == torch.float64:
            graph.x = graph.x.float()

        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)

        return out

    def reset(self) -> None:
        """Reset model parameters with consistent initialization and memory cleanup."""
        if self.model is not None:
            # Clear old model from memory first
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Rebuild model which will use the current random seed state
            self._build_model()


class GCNModel(nn.Module):
    """Simple 2-layer GCN model for node classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Ensure float32 for compatibility 
        if x.dtype == torch.float64:
            x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GATDownstreamModel(DownstreamModel):
    """
    GAT model for downstream tasks.
    
    Uses Graph Attention Networks for node classification.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: Optional[int] = None,
                 heads: int = 1,
                 device: Optional[torch.device] = None):
        """Initialize GAT model."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.device = device if device is not None else torch.device('cpu')
        
        self.model = None
        self.optimizer = None
        self._build_model()
    
    def _build_model(self):
        """Build the GAT model."""
        if self.output_dim is None:
            return
            
        self.model = GATModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            heads=self.heads
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )
    
    def train_model(self,
                    graph: Data,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    labels: torch.Tensor,
                    epochs: int = 100) -> None:
        """Train the GAT model with memory optimization."""
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        self.model.train()

        # Move to device and ensure proper dtype
        if graph.x.dtype == torch.float64:
            graph.x = graph.x.float()
        graph = graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        labels = labels.to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            try:
                out = self.model(graph.x, graph.edge_index)
                loss = F.nll_loss(out[train_mask], labels[train_mask])

                loss.backward()
                self.optimizer.step()

                # Memory cleanup every 10 epochs
                if epoch % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ GAT Memory error at epoch {epoch}, clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    def evaluate(self,
                 graph: Data,
                 test_mask: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """Evaluate GAT model."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            test_loss = F.nll_loss(out[test_mask], labels[test_mask].to(out.device))
        return float(test_loss)

    def predict(self, graph: Data) -> torch.Tensor:
        """Generate predictions for all nodes in the graph."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
        return out

    def reset(self) -> None:
        """Reset GAT model with consistent initialization and memory cleanup."""
        if self.model is not None:
            # Clear old model from memory first
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Rebuild model which will use the current random seed state
            self._build_model()


class GATModel(nn.Module):
    """2-layer GAT model for node classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, heads: int = 1):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6)
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)