"""
统一的基准测试框架

使用模型注册机制，支持开发模型和baseline模型的统一测试和比较。
"""

import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
import gc
import psutil
import os

from ..models import model_registry, GraphSummarizationModel, DownstreamModel
from ..datasets import DatasetLoader
from ..metrics import ComplexityMetric, InformationMetric, AccuracyMetric, SNRAnalysis, ICAnalysis


class UnifiedBenchmark:
    """
    统一基准测试类

    支持通过模型名称自动创建和测试任意Graph Summarization模型。
    包含内存优化和CUDA内存管理功能。
    """

    def __init__(self,
                 results_dir: str = './results/unified_benchmark',
                 device: str = 'cuda',
                 data_dir: str = './data',
                 random_seed: int = 42,
                 memory_monitor: bool = True):
        """
        初始化统一基准测试

        Args:
            results_dir: 结果保存目录
            device: 计算设备
            data_dir: 数据目录
            random_seed: 随机种子，确保下游任务模型初始化一致
            memory_monitor: 是否启用内存监控
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir  # 保存data_dir以便后续使用
        self.dataset_loader = DatasetLoader(data_dir)
        self.random_seed = random_seed
        self.memory_monitor = memory_monitor

        # 初始化度量
        self.complexity_metric = ComplexityMetric()

        # 实验设置追踪
        self.experiment_configs = {}  # 存储实验配置

        # Memory management settings
        self.enable_memory_optimization = True
        if torch.cuda.is_available():
            # Set CUDA memory allocation configuration
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    def _get_experiment_id(self, dataset_name: str, task_type: str, downstream_model: str) -> str:
        """生成实验设置ID"""
        return f"{dataset_name}_{task_type}_{downstream_model}"

    def _create_experiment_structure(self, dataset_name: str, task_type: str, downstream_model: str) -> Path:
        """创建实验结果的目录结构"""
        exp_id = self._get_experiment_id(dataset_name, task_type, downstream_model)

        # 创建实验目录结构
        exp_dir = self.results_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (exp_dir / "process_results").mkdir(exist_ok=True)
        (exp_dir / "comprehensive_results").mkdir(exist_ok=True)
        (exp_dir / "summary_graphs").mkdir(exist_ok=True)
        (exp_dir / "graph_visualizations").mkdir(exist_ok=True)
        (exp_dir / "training_curves").mkdir(exist_ok=True)

        # 记录实验配置
        self.experiment_configs[exp_id] = {
            'dataset': dataset_name,
            'task_type': task_type,
            'downstream_model': downstream_model,
            'experiment_dir': exp_dir,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }

        return exp_dir
        
    def run_single_model(self,
                        model_name: str,
                        dataset_name: str = 'Cora',
                        task_type: str = 'original',
                        downstream_model: str = 'gcn',
                        num_steps: int = 10,
                        epochs: int = 100,
                        model_kwargs: Dict = None,
                        downstream_kwargs: Dict = None) -> Dict[str, Any]:
        """
        测试单个模型的性能

        Args:
            model_name: 注册的模型名称
            dataset_name: 数据集名称
            task_type: 标签任务类型 ('original' 或 'degree')
            downstream_model: 下游任务模型类型
            num_steps: 图总结步数
            epochs: 下游任务训练轮数
            model_kwargs: 模型初始化参数
            downstream_kwargs: 下游任务模型参数

        Returns:
            包含测试结果的字典
        """
        print(f"\n{'='*80}")
        print(f"测试模型: {model_name} on {dataset_name} ({task_type} task)")
        print(f"{'='*80}")

        # 创建实验目录结构
        exp_dir = self._create_experiment_structure(dataset_name, task_type, downstream_model)
        exp_id = self._get_experiment_id(dataset_name, task_type, downstream_model)

        # 参数默认值
        model_kwargs = model_kwargs or {}
        downstream_kwargs = downstream_kwargs or {}
        
        # 加载数据集
        print(f"加载数据集 {dataset_name} with {task_type} task...")

        # Handle PPI multi-label tasks
        if dataset_name == 'PPI' and task_type == 'original':
            # Extract PPI label index from kwargs if provided
            ppi_label_index = model_kwargs.pop('ppi_label_index', 0)  # Remove from model_kwargs
            original_graph, train_mask, val_mask, test_mask = self.dataset_loader.load_dataset(
                dataset_name, task_type=task_type, ppi_label_index=ppi_label_index)
        else:
            original_graph, train_mask, val_mask, test_mask = self.dataset_loader.load_dataset(
                dataset_name, task_type=task_type)

        original_graph = self.dataset_loader.preprocess_for_summarization(original_graph)
        
        input_dim = original_graph.x.size(1)
        print(f"图: {original_graph.num_nodes}节点, {original_graph.edge_index.shape[1]}边")
        print(f"特征维度: {input_dim}")
        
        # 创建图总结模型
        print(f"创建图总结模型: {model_name}")
        try:
            # 获取模型信息
            model_info = model_registry.get_model_info(model_name)
            print(f"模型类别: {model_info['category']}")
            print(f"描述: {model_info['description']}")
            
            # 自动设置需要input_dim的模型参数（所有开发模型都需要）
            if model_info['category'] == 'development':
                if 'input_dim' not in model_kwargs:
                    model_kwargs['input_dim'] = input_dim
                    print(f"  自动设置input_dim = {input_dim}")
                if 'device' not in model_kwargs:
                    model_kwargs['device'] = self.device
            
            # 创建模型实例
            gs_model = model_registry.create_model(model_name, **model_kwargs)
            
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
            return {'error': str(e)}
        
        # 创建下游任务模型
        print(f"创建下游任务模型: {downstream_model}")
        if downstream_model.lower() == 'gcn':
            from ..models import GCNDownstreamModel
            dt_model = GCNDownstreamModel(
                input_dim=input_dim, 
                device=self.device,
                **downstream_kwargs
            )
        elif downstream_model.lower() == 'gat':
            from ..models import GATDownstreamModel
            dt_model = GATDownstreamModel(
                input_dim=input_dim,
                device=self.device, 
                **downstream_kwargs
            )
        else:
            raise ValueError(f"不支持的下游任务模型: {downstream_model}")
        
        # 训练模型（如果是开发模型）
        training_time = 0
        training_history = None
        if model_info['category'] == 'development':
            print(f"🚀 训练{model_name}模型 ...")
            print(f"📋 训练和测试将使用相同的下游任务模型: {downstream_model}")
            start_time = time.time()
            
            try:
                # 检查是否是Neural-Enhanced模型，需要特殊处理
                if 'neural_enhanced' in model_name:
                    # Neural-Enhanced模型有自己的包装器
                    from ..models.neural_enhanced_gradient import TrainableNeuralEnhancedGradientModel

                    # 设置训练数据
                    gs_model.set_training_data(train_mask, val_mask, original_graph.y)

                    # 创建可训练包装器
                    trainable_model = TrainableNeuralEnhancedGradientModel(
                        model=gs_model,
                        training_strategy='fixed_uniform'
                    )

                    # 训练模型
                    training_history = trainable_model.train_model(
                        graph=original_graph,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        labels=original_graph.y,
                        epochs=epochs,  # 使用传入的epochs参数
                        num_steps=num_steps,
                        downstream_epochs=20  # 下游模型训练轮数
                    )

                    # 使用包装器的模型
                    gs_model = trainable_model.model

                else:
                    # 普通开发模型使用原有的训练逻辑
                    from ..models.training_variants import TrainableGraphSummarizationModel

                    # 创建下游任务模型工厂函数，确保训练和测试使用相同的模型
                    def downstream_model_factory():
                        # 复制当前的下游任务模型配置
                        if downstream_model.lower() == 'gcn':
                            from ..models import GCNDownstreamModel
                            return GCNDownstreamModel(
                                input_dim=input_dim,
                                device=self.device,
                                **downstream_kwargs
                            )
                        elif downstream_model.lower() == 'gat':
                            from ..models import GATDownstreamModel
                            return GATDownstreamModel(
                                input_dim=input_dim,
                                device=self.device,
                                **downstream_kwargs
                            )
                        else:
                            raise ValueError(f"不支持的下游任务模型: {downstream_model}")

                    # 包装模型使其可训练，传入相同的下游模型工厂
                    trainable_model = TrainableGraphSummarizationModel(
                        model=gs_model,
                        downstream_model_factory=downstream_model_factory
                    )

                    # 训练模型
                    training_history = trainable_model.train(
                        graph=original_graph,
                        train_labels=original_graph.y,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        num_epochs=50,  # Reduced for benchmark efficiency
                        num_steps=num_steps
                    )

                    # 更新gs_model为训练后的模型
                    gs_model = trainable_model.model
                
                training_time = time.time() - start_time
                print(f"✅ 模型训练完成，耗时: {training_time:.2f}s")
                
            except Exception as e:
                print(f"❌ 模型训练失败: {e}")
                return {'error': f'Training failed: {e}'}
        
        # 生成总结图
        print(f"📊 生成{num_steps+1}个总结图...")
        self._log_memory_usage("Before graph summarization")

        start_time = time.time()
        try:
            # Clear CUDA cache before summarization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            summary_graphs = gs_model.summarize(original_graph, num_steps=num_steps)
            summarization_time = time.time() - start_time
            print(f"✅ 总结生成完成，耗时: {summarization_time:.2f}s")

            self._log_memory_usage("After graph summarization")

        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"❌ 图总结生成失败: CUDA内存不足")
                print(f"尝试减少步数或使用CPU...")
                # Try with CPU if CUDA OOM
                if self.device.type == 'cuda':
                    cpu_device = torch.device('cpu')
                    original_graph = original_graph.to(cpu_device)
                    gs_model.device = cpu_device
                    if hasattr(gs_model, 'model') and gs_model.model is not None:
                        gs_model.model = gs_model.model.to(cpu_device)
                    summary_graphs = gs_model.summarize(original_graph, num_steps=num_steps)
                    summarization_time = time.time() - start_time
                    print(f"✅ CPU总结生成完成，耗时: {summarization_time:.2f}s")
                else:
                    raise e
            else:
                print(f"❌ 图总结生成失败: {e}")
                return {'error': f'Summarization failed: {e}'}
        
        # 打印边数统计
        print("总结图边数:")
        for i, graph in enumerate(summary_graphs):
            print(f"  步骤 {i}: {graph.edge_index.shape[1]} 边")
        
        # 计算度量指标
        print("📈 计算度量指标...")
        complexity_metrics = self.complexity_metric.compute_list(summary_graphs, original_graph)

        self._log_memory_usage("Before information metric computation")

        # 使用固定的随机种子初始化InformationMetric，确保下游模型初始化一致
        info_metric = InformationMetric(dt_model, self.device, random_seed=self.random_seed)
        print(f"🧠 训练{len(summary_graphs)}个下游任务模型进行信息度量...")

        # Reduce epochs for memory efficiency if needed
        adaptive_epochs = self._get_adaptive_epochs(epochs, len(summary_graphs))
        if adaptive_epochs != epochs:
            print(f"⚡ 为节约内存，减少训练轮数: {epochs} -> {adaptive_epochs}")

        # 计算两种归一化的信息度量
        info_metrics_log = None
        info_metrics_add = None

        try:
            print("📊 计算对数比率归一化信息度量...")
            info_metrics_log = info_metric.compute_list(
                summary_graphs, train_mask, val_mask, test_mask,
                original_graph.y, epochs=adaptive_epochs, normalization='log_ratio'
            )

            print("📊 计算加法归一化信息度量...")
            # Reset model for different normalization calculation
            info_metric_add = InformationMetric(dt_model, self.device, random_seed=self.random_seed)
            info_metrics_add = info_metric_add.compute_list(
                summary_graphs, train_mask, val_mask, test_mask,
                original_graph.y, epochs=adaptive_epochs, normalization='additive'
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ 信息度量计算失败: CUDA内存不足")
                print(f"尝试进一步减少训练轮数...")
                # Try with even fewer epochs
                emergency_epochs = max(adaptive_epochs // 2, 10)
                print(f"🚨 紧急模式: 使用 {emergency_epochs} 轮训练")

                info_metrics_log = info_metric.compute_list(
                    summary_graphs, train_mask, val_mask, test_mask,
                    original_graph.y, epochs=emergency_epochs, normalization='log_ratio'
                )

                info_metric_add = InformationMetric(dt_model, self.device, random_seed=self.random_seed)
                info_metrics_add = info_metric_add.compute_list(
                    summary_graphs, train_mask, val_mask, test_mask,
                    original_graph.y, epochs=emergency_epochs, normalization='additive'
                )
            else:
                raise e

        self._log_memory_usage("After information metric computation")

        # 计算Accuracy度量
        print("🎯 计算Accuracy度量...")
        accuracy_metrics = None
        try:
            accuracy_metric = AccuracyMetric(dt_model, self.device, random_seed=self.random_seed)
            accuracy_metrics = accuracy_metric.compute_list(
                summary_graphs, train_mask, val_mask, test_mask,
                original_graph.y, epochs=adaptive_epochs
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Accuracy度量计算失败: CUDA内存不足")
                print(f"尝试进一步减少训练轮数...")
                emergency_epochs = max(adaptive_epochs // 2, 10)
                print(f"🚨 紧急模式: 使用 {emergency_epochs} 轮训练")

                accuracy_metric = AccuracyMetric(dt_model, self.device, random_seed=self.random_seed)
                accuracy_metrics = accuracy_metric.compute_list(
                    summary_graphs, train_mask, val_mask, test_mask,
                    original_graph.y, epochs=emergency_epochs
                )
            else:
                raise e

        self._log_memory_usage("After accuracy metric computation")

        # 计算IC-AUC和信息阈值点 (使用两种归一化)
        from ..metrics import ICAnalysis

        ic_auc_log = ICAnalysis.compute_ic_auc(complexity_metrics, info_metrics_log)
        ic_auc_add = ICAnalysis.compute_ic_auc(complexity_metrics, info_metrics_add)

        threshold_point_log = ICAnalysis.compute_information_threshold_point(
            complexity_metrics, info_metrics_log, threshold=0.8)
        threshold_point_add = ICAnalysis.compute_information_threshold_point(
            complexity_metrics, info_metrics_add, threshold=0.8)

        # 保持向后兼容
        snr_auc = ic_auc_add  # 默认使用加法归一化
        
        # 整理结果
        result = {
            'model_name': model_name,
            'model_info': model_info,
            'dataset': dataset_name,
            'task_type': task_type,
            'downstream_model': downstream_model,
            'num_steps': num_steps,
            'epochs': epochs,
            'complexity_metrics': complexity_metrics,
            # 双重归一化信息度量
            'information_metrics_log_ratio': info_metrics_log,
            'information_metrics_additive': info_metrics_add,
            # Accuracy度量
            'accuracy_metrics': accuracy_metrics,
            # IC-AUC指标（两种归一化）
            'ic_auc_log_ratio': ic_auc_log,
            'ic_auc_additive': ic_auc_add,
            # 信息阈值点（两种归一化）
            'threshold_point_log_ratio': threshold_point_log,
            'threshold_point_additive': threshold_point_add,
            # 向后兼容的字段
            'information_metrics': info_metrics_add,  # 默认使用加法归一化
            'snr_auc': snr_auc,  # 向后兼容
            'training_time': training_time,
            'summarization_time': summarization_time,
            'training_history': training_history,
            'summary_graphs': summary_graphs,
            'original_graph': original_graph,
            'exp_dir': exp_dir,
            'exp_id': exp_id,
            'success': True
        }

        # 保存过程结果
        self._save_process_results(result)

        # 打印结果摘要
        print(f"\n{'='*80}")
        print("测试结果")
        print(f"{'='*80}")
        print(f"模型: {model_name} ({model_info['category']})")
        print(f"SNR-AUC: {snr_auc:.4f}")
        print(f"复杂度指标: {[f'{x:.0f}' for x in complexity_metrics]}")
        print(f"信息指标(log): {[f'{x:.4f}' for x in info_metrics_log]}")
        print(f"信息指标(add): {[f'{x:.4f}' for x in info_metrics_add]}")
        if accuracy_metrics is not None:
            print(f"准确度指标: {[f'{x:.4f}' for x in accuracy_metrics]}")

        # 生成单个模型的SNR曲线图（保存到实验目录）
        self._plot_single_model_enhanced(result)
        
        # Final memory cleanup
        if self.enable_memory_optimization:
            self._cleanup_memory()

        return result
    
    def compare_models(self,
                      model_names: List[str],
                      dataset_name: str = 'Cora',
                      task_type: str = 'original',
                      downstream_model: str = 'gcn',
                      num_steps: int = 10,
                      epochs: int = 100) -> Dict[str, Any]:
        """
        比较多个模型的性能

        Args:
            model_names: 模型名称列表
            dataset_name: 数据集名称
            task_type: 标签任务类型 ('original' 或 'degree')
            downstream_model: 下游任务模型
            num_steps: 图总结步数
            epochs: 训练轮数

        Returns:
            包含所有模型结果的字典
        """
        print(f"\n{'='*100}")
        print(f"模型比较: {len(model_names)} 个模型 on {dataset_name} ({task_type} task)")
        print(f"{'='*100}")
        print(f"模型列表: {', '.join(model_names)}")
        
        all_results = {}
        successful_results = []
        
        for model_name in model_names:
            try:
                # 对于需要特殊参数的模型，自动配置
                model_kwargs = {}
                
                result = self.run_single_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    downstream_model=downstream_model,
                    num_steps=num_steps,
                    epochs=epochs,
                    model_kwargs=model_kwargs
                )
                
                all_results[model_name] = result
                if result.get('success', False):
                    successful_results.append(result)
                    
            except Exception as e:
                print(f"❌ 模型 {model_name} 测试失败: {e}")
                all_results[model_name] = {'error': str(e), 'success': False}
        
        if not successful_results:
            print("❌ 没有成功的测试结果")
            return all_results
        
        # 生成综合结果
        if successful_results:
            # 从第一个成功结果获取实验配置
            first_result = successful_results[0]
            exp_id = first_result['exp_id']
            exp_dir = first_result['exp_dir']

            # 保存综合结果
            self._save_comprehensive_results(successful_results, exp_dir)

        # 保存详细结果（兼容原有功能）
        self._save_detailed_results(successful_results, dataset_name, downstream_model)

        # 生成对比报告
        self._generate_comparison_report(successful_results, dataset_name, downstream_model)

        # 绘制对比图
        self._plot_comparison(successful_results, dataset_name, downstream_model)
        
        return all_results

    def _save_process_results(self, result: Dict):
        """保存单个模型的过程结果"""
        exp_dir = result['exp_dir']
        model_name = result['model_name']

        # 1. 保存简化过程中每一步的度量数据 (TSV格式)
        self._save_step_metrics(result)

        # 2. 保存多步简化图 (CSV格式 + 可视化)
        self._save_summary_graphs(result)

        # 3. 保存训练曲线 (如果有训练过程)
        if result['training_history'] is not None:
            self._save_training_curves(result)

        print(f"✅ 过程结果已保存到: {exp_dir}")

    def _save_step_metrics(self, result: Dict):
        """保存每步的复杂度和信息度量数据（支持双重归一化）"""
        exp_dir = result['exp_dir']
        model_name = result['model_name']

        # 检查必要的字段是否存在
        complexity_metrics = result.get('complexity_metrics', [])
        info_metrics_log = result.get('information_metrics_log_ratio', [])
        info_metrics_add = result.get('information_metrics_additive', [])
        accuracy_metrics = result.get('accuracy_metrics', [])

        # 向后兼容：如果没有双重归一化数据，使用旧字段
        if not info_metrics_log and not info_metrics_add:
            info_metrics_old = result.get('information_metrics', [])
            if info_metrics_old:
                print(f"⚠️ {model_name}: 使用向后兼容模式，将单一信息度量复制为双重数据")
                info_metrics_log = info_metrics_old
                info_metrics_add = info_metrics_old

        if not complexity_metrics or not info_metrics_log or not info_metrics_add:
            print(f"❌ {model_name}: 缺少必要的度量数据")
            print(f"   complexity_metrics: {len(complexity_metrics)} 项")
            print(f"   info_metrics_log_ratio: {len(info_metrics_log)} 项")
            print(f"   info_metrics_additive: {len(info_metrics_add)} 项")
            print(f"   accuracy_metrics: {len(accuracy_metrics)} 项")
            return

        # 创建步骤度量数据
        step_data = []

        # 准备数据进行zip操作
        zip_data = [complexity_metrics, info_metrics_log, info_metrics_add]
        zip_names = ['complexity', 'info_log', 'info_add']

        # 如果有accuracy数据，添加到zip中
        if accuracy_metrics and len(accuracy_metrics) == len(complexity_metrics):
            zip_data.append(accuracy_metrics)
            zip_names.append('accuracy')

        for i, values in enumerate(zip(*zip_data)):
            step_dict = {
                'step': i,
                'complexity_metric': values[0],  # complexity
                'information_metric_log_ratio': values[1],  # info_log
                'information_metric_additive': values[2],  # info_add
                'model': model_name,
                'dataset': result['dataset'],
                'task_type': result['task_type'],
                'downstream_model': result['downstream_model']
            }

            # 如果有accuracy数据，添加到字典中
            if len(values) > 3:
                step_dict['accuracy_metric'] = values[3]

            step_data.append(step_dict)

        # 保存为TSV格式
        df = pd.DataFrame(step_data)
        tsv_path = exp_dir / "process_results" / f"{model_name}_step_metrics.tsv"
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"📊 步骤度量数据保存到: {tsv_path}")
        if accuracy_metrics:
            print(f"   包含 {len(step_data)} 步数据，双重归一化Information Measure + Accuracy Metric")
        else:
            print(f"   包含 {len(step_data)} 步数据，双重归一化Information Measure")

    def _save_summary_graphs(self, result: Dict):
        """保存简化图的稀疏表示和可视化"""
        exp_dir = result['exp_dir']
        model_name = result['model_name']
        summary_graphs = result['summary_graphs']
        dataset_name = result.get('dataset', '')

        # 0. 保存节点信息（标签和数据集划分）- 每个实验设置只保存一次
        self._save_node_info(result)

        # 1. 保存稀疏图 (CSV格式)
        for step, graph in enumerate(summary_graphs):
            # 检查是否为SO_relation数据集，如果是则使用特殊格式
            if dataset_name.startswith('SO_relation'):
                self._save_so_relation_format(graph, exp_dir, model_name, step, result)
            else:
                # 标准格式：保存边列表
                edge_list = graph.edge_index.t().cpu().numpy()
                edge_df = pd.DataFrame(edge_list, columns=['source', 'target'])
                edge_df['step'] = step
                edge_df['model'] = model_name

                csv_path = exp_dir / "summary_graphs" / f"{model_name}_step_{step}_edges.csv"
                edge_df.to_csv(csv_path, index=False)

        print(f"📁 简化图CSV文件保存到: {exp_dir / 'summary_graphs'}")

        # 2. 生成可视化图 (PNG格式, 无中文)
        self._visualize_summary_graphs(result)

    def _save_so_relation_format(self, graph: Data, exp_dir: Path, model_name: str, step: int, result: Dict):
        """为SO_relation数据集保存特殊格式的简化图"""
        original_graph = result['original_graph']
        dataset_name = result['dataset']

        # 获取原始图的KO映射信息
        if hasattr(original_graph, '_idx_to_ko'):
            idx_to_ko = original_graph._idx_to_ko
        else:
            print(f"⚠️ 原始图缺少KO映射信息，无法保存SO_relation格式")
            return

        # 获取边列表
        edge_index = graph.edge_index.cpu()
        edge_list = []

        # 转换为KO格式的边列表
        for i in range(edge_index.size(1)):
            src_idx = edge_index[0, i].item()
            tgt_idx = edge_index[1, i].item()

            # 转换索引为KO ID
            if src_idx in idx_to_ko and tgt_idx in idx_to_ko:
                src_ko = idx_to_ko[src_idx]
                tgt_ko = idx_to_ko[tgt_idx]

                # 避免重复边（无向图）
                if src_ko < tgt_ko:  # 按字典序排序避免重复
                    edge_list.append((src_ko, tgt_ko, 1.0))  # 权重设为1.0（简化图为无权图）

        # 创建DataFrame，格式与原始数据一致
        if edge_list:
            edge_df = pd.DataFrame(edge_list, columns=['KO1', 'KO2', 'Weight'])
        else:
            # 如果没有边，创建空的DataFrame但保持格式
            edge_df = pd.DataFrame(columns=['KO1', 'KO2', 'Weight'])

        # 保存为TSV格式（与原始数据格式一致）
        tsv_path = exp_dir / "summary_graphs" / f"{model_name}_step_{step}_ko_relation.tsv"
        edge_df.to_csv(tsv_path, sep='\t', index=False)

        print(f"💾 SO_relation格式保存: {tsv_path} ({len(edge_df)} edges)")

        # 同时保存标准格式以供其他功能使用
        edge_list_std = graph.edge_index.t().cpu().numpy()
        edge_df_std = pd.DataFrame(edge_list_std, columns=['source', 'target'])
        edge_df_std['step'] = step
        edge_df_std['model'] = model_name

        csv_path = exp_dir / "summary_graphs" / f"{model_name}_step_{step}_edges.csv"
        edge_df_std.to_csv(csv_path, index=False)

    def _save_node_info(self, result: Dict):
        """保存节点标签和数据集划分信息（每个实验设置只保存一次）"""
        exp_dir = result['exp_dir']
        model_name = result['model_name']
        original_graph = result['original_graph']

        # 检查文件是否已存在（因为同一个实验设置的所有模型共享相同的节点信息）
        node_info_path = exp_dir / "summary_graphs" / "node_info.csv"
        if node_info_path.exists():
            return  # 已经保存过，无需重复保存

        # 获取节点标签
        if not hasattr(original_graph, 'y') or original_graph.y is None:
            print(f"⚠️ 原始图缺少标签信息，跳过节点信息保存")
            return

        labels = original_graph.y.cpu().numpy()
        num_nodes = original_graph.num_nodes

        # 从result中获取train/val/test mask（这些是在run_single_model中保存的）
        # 注意：这些信息存储在实验目录级别，而不是在result中
        # 我们需要从原始的数据集加载中获取这些mask
        # 简化处理：我们在这里重新加载数据集以获取mask
        try:
            from ..datasets import DatasetLoader
            dataset_loader = DatasetLoader(self.data_dir)
            dataset_name = result['dataset']
            task_type = result['task_type']

            # 重新加载数据集以获取mask
            _, train_mask, val_mask, test_mask = dataset_loader.load_dataset(
                dataset_name, task_type=task_type
            )

            # 转换mask为字符串标签
            split_labels = []
            for i in range(num_nodes):
                if train_mask[i]:
                    split_labels.append('train')
                elif val_mask[i]:
                    split_labels.append('val')
                elif test_mask[i]:
                    split_labels.append('test')
                else:
                    split_labels.append('unlabeled')  # 如果存在未标记的节点

        except Exception as e:
            print(f"⚠️ 无法获取数据集划分信息: {e}")
            # 如果无法获取mask，使用默认值
            split_labels = ['unknown'] * num_nodes

        # 创建节点信息DataFrame
        node_data = {
            'node_id': list(range(num_nodes)),
            'label': labels.tolist(),
            'split': split_labels
        }

        node_df = pd.DataFrame(node_data)

        # 保存为CSV
        node_df.to_csv(node_info_path, index=False)
        print(f"💾 节点信息保存到: {node_info_path} ({num_nodes} nodes)")

    def _visualize_summary_graphs(self, result: Dict):
        """可视化简化图序列，使用不同颜色标注不同标签的节点，显示accuracy"""
        exp_dir = result['exp_dir']
        model_name = result['model_name']
        summary_graphs = result['summary_graphs']
        original_graph = result['original_graph']

        # 如果节点数目很多，跳过可视化
        if original_graph.num_nodes > 200:
            print(f"⚠️ 节点数目过多 ({original_graph.num_nodes} > 200)，跳过图可视化")
            return

        # 获取节点标签
        if not hasattr(original_graph, 'y') or original_graph.y is None:
            print(f"⚠️ 原始图缺少标签信息，跳过图可视化")
            return

        node_labels = original_graph.y.cpu().numpy()
        unique_labels = np.unique(node_labels)
        num_labels = len(unique_labels)

        # 为不同标签分配颜色
        if num_labels <= 10:
            cmap = plt.cm.tab10
        elif num_labels <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.hsv

        label_colors = {}
        for idx, label in enumerate(unique_labels):
            label_colors[label] = cmap(idx % cmap.N)

        # 获取accuracy metrics（如果存在）
        accuracy_metrics = result.get('accuracy_metrics', None)

        # 基于原始图计算固定的节点位置
        try:
            original_nx = to_networkx(original_graph, to_undirected=True)

            # 使用固定的随机种子确保布局一致性
            np.random.seed(42)
            fixed_pos = nx.spring_layout(original_nx, k=1, iterations=100, seed=42)

        except Exception as e:
            print(f"⚠️ 无法计算固定布局: {e}")
            return

        # 对每一步都进行可视化
        num_steps = len(summary_graphs)
        key_steps = list(range(num_steps))

        for step in key_steps:
            graph = summary_graphs[step]

            try:
                nx_graph = to_networkx(graph, to_undirected=True)

                plt.figure(figsize=(12, 9))

                # 使用固定位置
                current_nodes = set(nx_graph.nodes())

                # 准备按标签分组的节点
                all_nodes = set(fixed_pos.keys())
                connected_nodes = current_nodes
                isolated_nodes = all_nodes - connected_nodes

                # 绘制有连接的节点（按标签着色）
                for label in unique_labels:
                    # 找到属于该标签且有连接的节点
                    label_connected_nodes = [
                        node for node in connected_nodes
                        if node < len(node_labels) and node_labels[node] == label
                    ]

                    if label_connected_nodes:
                        label_pos = {node: fixed_pos[node] for node in label_connected_nodes}
                        nx.draw_networkx_nodes(
                            nx_graph, label_pos,
                            nodelist=label_connected_nodes,
                            node_size=50,
                            node_color=[label_colors[label]],
                            alpha=0.8,
                            label=f'Label {int(label)}'
                        )

                # 绘制孤立节点（按标签着色，但更透明更小）
                for label in unique_labels:
                    label_isolated_nodes = [
                        node for node in isolated_nodes
                        if node < len(node_labels) and node_labels[node] == label
                    ]

                    if label_isolated_nodes:
                        label_pos = {node: fixed_pos[node] for node in label_isolated_nodes}
                        nx.draw_networkx_nodes(
                            nx_graph, label_pos,
                            nodelist=label_isolated_nodes,
                            node_size=20,
                            node_color=[label_colors[label]],
                            alpha=0.3
                        )

                # 绘制边
                if nx_graph.number_of_edges() > 0:
                    current_pos = {node: pos for node, pos in fixed_pos.items() if node in current_nodes}
                    nx.draw_networkx_edges(nx_graph, current_pos, alpha=0.4, width=0.8)

                # 构建标题（包含accuracy信息）
                title = f'Step {step}: {original_graph.num_nodes} nodes, {nx_graph.number_of_edges()} edges'
                if accuracy_metrics is not None and step < len(accuracy_metrics):
                    accuracy = accuracy_metrics[step]
                    title += f', Accuracy: {accuracy:.4f}'

                plt.title(title, fontsize=14)
                plt.axis('off')

                # 添加图例（显示标签颜色）
                if num_labels <= 10:  # 只在标签数量不太多时显示图例
                    plt.legend(loc='upper right', fontsize=9, framealpha=0.9)

                # 保存图片
                png_path = exp_dir / "graph_visualizations" / f"{model_name}_step_{step}_graph.png"
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"⚠️ 步骤 {step} 可视化失败: {e}")
                import traceback
                traceback.print_exc()

        print(f"🖼️ 图可视化保存到: {exp_dir / 'graph_visualizations'} (按标签着色，显示accuracy)")

    def _save_training_curves(self, result: Dict):
        """保存训练曲线"""
        exp_dir = result['exp_dir']
        model_name = result['model_name']
        training_history = result['training_history']

        if training_history is None:
            return

        try:
            # 绘制训练曲线
            plt.figure(figsize=(12, 4))

            if isinstance(training_history, dict):
                # 处理不同类型的训练历史格式
                if 'train_loss' in training_history:
                    epochs = range(1, len(training_history['train_loss']) + 1)

                    plt.subplot(1, 2, 1)
                    plt.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss')
                    if 'val_loss' in training_history:
                        plt.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training Loss Curve')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    plt.subplot(1, 2, 2)
                    if 'train_acc' in training_history:
                        plt.plot(epochs, training_history['train_acc'], 'b-', label='Train Accuracy')
                    if 'val_acc' in training_history:
                        plt.plot(epochs, training_history['val_acc'], 'r-', label='Validation Accuracy')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.title('Training Accuracy Curve')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()

                    # 保存训练曲线图
                    curve_path = exp_dir / "training_curves" / f"{model_name}_training_curves.png"
                    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"📈 训练曲线保存到: {curve_path}")

        except Exception as e:
            print(f"⚠️ 训练曲线保存失败: {e}")

    def _plot_single_model_enhanced(self, result: Dict):
        """生成增强版单模型IC曲线图"""
        exp_dir = result['exp_dir']
        model_name = result['model_name']
        model_info = result['model_info']
        complexity_metrics = result['complexity_metrics']
        information_metrics = result['information_metrics_additive']  # 使用加法归一化

        plt.figure(figsize=(10, 6))

        # 使用现有的图标系统
        development_markers = {
            'gradient_based': ('o', '#1f77b4'),
            'neural_enhanced_main': ('s', '#ff7f0e'),
            'neural_enhanced_high_fusion': ('^', '#2ca02c'),
            'neural_enhanced_low_fusion': ('v', '#d62728'),
            'neural_enhanced_no_residual': ('D', '#9467bd'),
            'neural_enhanced_slow_gradient': ('*', '#8c564b'),
        }

        baseline_markers = {
            'networkit_forest_fire': ('o', '#e377c2'),
            'networkit_local_degree': ('s', '#ff7f0e'),
            'networkit_local_similarity': ('^', '#ffbb78'),
            'networkit_random_edge': ('v', '#2ECC71'),
            'networkit_random_node_edge': ('D', '#3498DB'),
            'networkit_scan': ('*', '#9467bd'),
            'networkit_simmelian': ('x', '#e377c2'),
            'pri_graphs': ('h', '#8c564b'),
        }

        # 获取模型的图标和颜色
        if model_info['category'] == 'development':
            if model_name in development_markers:
                marker, color = development_markers[model_name]
            else:
                marker, color = 'o', '#1f77b4'  # Default: circle, blue
            linestyle = '-'
        elif model_info['category'] == 'baseline':
            if model_name in baseline_markers:
                marker, color = baseline_markers[model_name]
            else:
                marker, color = 's', '#E74C3C'  # Default: square, red
            linestyle = '--'
        else:
            marker, color = '^', 'green'  # Default: triangle, green
            linestyle = ':'

        # 绘制IC曲线
        plt.plot(
            complexity_metrics,
            information_metrics,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{model_name} (IC-AUC: {result['ic_auc_additive']:.3f})",
            linestyle=linestyle
        )

        plt.xlabel('Complexity Metric (Normalized Edge Count)', fontsize=12)
        plt.ylabel('Information Metric (Additive Normalization)', fontsize=12)
        plt.title(f'IC Curve - {model_name}\n{result["dataset"]} + {result["downstream_model"].upper()} ({result["task_type"]} task)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # 保存到实验目录
        plot_path = exp_dir / "process_results" / f"{model_name}_ic_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 IC曲线图保存到: {plot_path}")

    def _save_comprehensive_results(self, results: List[Dict], exp_dir: Path):
        """保存综合结果"""
        # 1. 生成加法归一化IC曲线对比图（默认优先级）
        self._plot_comprehensive_ic_curves(results, exp_dir, 'additive')

        # 2. 生成对数比率归一化IC曲线对比图
        self._plot_comprehensive_ic_curves(results, exp_dir, 'log_ratio')

        # 3. 保存IC-AUC对比表 (TSV格式，包含两种归一化)
        self._save_ic_auc_table(results, exp_dir)

        # 4. 保存信息阈值点对比表
        self._save_threshold_point_table(results, exp_dir)

        print(f"✅ 综合结果已保存到: {exp_dir / 'comprehensive_results'}")

    def _plot_comprehensive_ic_curves(self, results: List[Dict], exp_dir: Path, normalization: str = 'log_ratio'):
        """绘制所有模型的IC曲线对比图"""
        plt.figure(figsize=(16, 12))

        # 选择信息度量数据
        if normalization == 'log_ratio':
            info_key = 'information_metrics_log_ratio'
            title_suffix = 'Log-ratio Normalization'
            ylabel = 'Information Metric (Log-ratio Normalization)'
        else:
            info_key = 'information_metrics_additive'
            title_suffix = 'Additive Normalization'
            ylabel = 'Information Metric (Additive Normalization)'

        # 使用更新的图标系统
        development_markers = {
            'gradient_based': ('o', '#1f77b4'),
            'neural_enhanced_main': ('s', '#ff7f0e'),
            'neural_enhanced_high_fusion': ('^', '#2ca02c'),
            'neural_enhanced_low_fusion': ('v', '#d62728'),
            'neural_enhanced_no_residual': ('D', '#9467bd'),
            'neural_enhanced_slow_gradient': ('*', '#8c564b'),
        }

        baseline_markers = {
            'networkit_forest_fire': ('o', '#e377c2'),
            'networkit_local_degree': ('s', '#ff7f0e'),
            'networkit_local_similarity': ('^', '#ffbb78'),
            'networkit_random_edge': ('v', '#2ECC71'),
            'networkit_random_node_edge': ('D', '#3498DB'),
            'networkit_scan': ('*', '#9467bd'),
            'networkit_simmelian': ('x', '#e377c2'),
            'pri_graphs': ('h', '#8c564b'),
        }

        for result in results:
            model_name = result['model_name']
            category = result['model_info']['category']

            # 获取对应归一化方式的AUC
            if normalization == 'log_ratio':
                ic_auc = result['ic_auc_log_ratio']
            else:
                ic_auc = result['ic_auc_additive']

            # 设置图标和颜色
            if category == 'development':
                if model_name in development_markers:
                    marker, color = development_markers[model_name]
                    linestyle = '-'
                else:
                    # Fallback for unknown development models
                    color = '#1f77b4'  # Default blue
                    marker = 'o'
                    linestyle = '-'

            elif category == 'baseline':
                if model_name in baseline_markers:
                    marker, color = baseline_markers[model_name]
                    linestyle = '--'
                else:
                    # Fallback for unknown baseline models
                    color = '#E74C3C'  # Default red
                    marker = 's'
                    linestyle = '--'
            else:
                # Unknown category
                color = 'green'
                marker = '^'
                linestyle = ':'

            plt.plot(
                result['complexity_metrics'],
                result[info_key],
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=8,
                alpha=0.8,
                label=f"{model_name} ({category}, AUC={ic_auc:.3f})"
            )

        # 获取实验配置信息（优先从experiment_configs，否则从结果中获取）
        if hasattr(self, 'experiment_configs') and result['exp_id'] in self.experiment_configs:
            exp_config = self.experiment_configs[result['exp_id']]
            dataset = exp_config["dataset"]
            downstream_model = exp_config["downstream_model"]
            task_type = exp_config["task_type"]
        else:
            # 从结果中直接获取
            dataset = result['dataset']
            downstream_model = result['downstream_model']
            task_type = result['task_type']

        plt.xlabel('Complexity Metric (Normalized Edge Count)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'IC Curve Comparison ({title_suffix}) - {dataset} + {downstream_model.upper()} ({task_type} task)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plot_path = exp_dir / "comprehensive_results" / f"ic_curves_comparison_{normalization}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 IC曲线对比图（{title_suffix}）保存到: {plot_path}")

    def _save_ic_auc_table(self, results: List[Dict], exp_dir: Path):
        """保存IC-AUC对比表（包含两种归一化）"""
        summary_data = []
        for result in results:
            summary_data.append({
                'model': result['model_name'],
                'category': result['model_info']['category'],
                'description': result['model_info']['description'],
                'ic_auc_log_ratio': result['ic_auc_log_ratio'],
                'ic_auc_additive': result['ic_auc_additive'],
                'threshold_point_log_ratio': result.get('threshold_point_log_ratio', None),
                'threshold_point_additive': result.get('threshold_point_additive', None),
                'training_time_seconds': result['training_time'],
                'summarization_time_seconds': result['summarization_time'],
                'dataset': result['dataset'],
                'task_type': result['task_type'],
                'downstream_model': result['downstream_model']
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('ic_auc_additive', ascending=False)

        tsv_path = exp_dir / "comprehensive_results" / "ic_auc_comparison.tsv"
        df_summary.to_csv(tsv_path, sep='\t', index=False)
        print(f"📋 IC-AUC对比表保存到: {tsv_path}")

    def _save_threshold_point_table(self, results: List[Dict], exp_dir: Path):
        """保存信息阈值点对比表"""
        threshold_data = []
        for result in results:
            threshold_data.append({
                'model': result['model_name'],
                'category': result['model_info']['category'],
                'threshold_point_log_ratio': result.get('threshold_point_log_ratio', None),
                'threshold_point_additive': result.get('threshold_point_additive', None),
                'dataset': result['dataset'],
                'task_type': result['task_type'],
                'downstream_model': result['downstream_model']
            })

        df_threshold = pd.DataFrame(threshold_data)
        # 按照阈值点排序（越小越好，None值排在最后）
        df_threshold['threshold_sort'] = df_threshold['threshold_point_log_ratio'].fillna(float('inf'))
        df_threshold = df_threshold.sort_values('threshold_sort', ascending=True)
        df_threshold = df_threshold.drop('threshold_sort', axis=1)

        tsv_path = exp_dir / "comprehensive_results" / "threshold_points_comparison.tsv"
        df_threshold.to_csv(tsv_path, sep='\t', index=False)
        print(f"📋 信息阈值点对比表保存到: {tsv_path}")

    def _save_detailed_results(self, results: List[Dict], dataset_name: str, downstream_model: str):
        """保存详细结果到CSV"""
        detailed_data = []
        for result in results:
            for i, (complexity, information) in enumerate(zip(
                result['complexity_metrics'], 
                result['information_metrics']
            )):
                detailed_data.append({
                    'model': result['model_name'],
                    'category': result['model_info']['category'],
                    'dataset': dataset_name,
                    'downstream': downstream_model,
                    'step': i,
                    'complexity': complexity,
                    'information': information,
                    'snr_auc': result['snr_auc']
                })
        
        df = pd.DataFrame(detailed_data)
        csv_path = self.results_dir / f'detailed_comparison_{dataset_name}_{downstream_model}.csv'
        df.to_csv(csv_path, index=False)
        print(f"详细结果保存到: {csv_path}")
    
    def _generate_comparison_report(self, results: List[Dict], dataset_name: str, downstream_model: str):
        """生成对比报告"""
        summary_data = []
        for result in results:
            summary_data.append({
                'model': result['model_name'],
                'category': result['model_info']['category'],
                'description': result['model_info']['description'],
                'snr_auc': result['snr_auc'],
                'time_seconds': result['summarization_time']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('snr_auc', ascending=False)
        
        tsv_path = self.results_dir / f'model_comparison_{dataset_name}_{downstream_model}.tsv'
        df_summary.to_csv(tsv_path, sep='\t', index=False)
        print(f"对比报告保存到: {tsv_path}")
        
        # 打印排名
        print(f"\n{'='*100}")
        print(f"{dataset_name} + {downstream_model.upper()} 模型性能排名")
        print(f"{'='*100}")
        print(f"{'排名':<4} {'模型':<20} {'类别':<12} {'SNR-AUC':<10} {'时间(s)':<8}")
        print("-" * 60)
        
        for i, row in df_summary.iterrows():
            print(f"{i+1:<4} {row['model']:<20} {row['category']:<12} {row['snr_auc']:<10.2f} {row['time_seconds']:<8.2f}")
    
    def _plot_comparison(self, results: List[Dict], dataset_name: str, downstream_model: str):
        """绘制对比图"""
        plt.figure(figsize=(16, 12))
        
        # Define distinct markers for all development models
        development_markers = {
            'learnable': ('o', '#1f77b4'),           # Circle, blue
            'learnable_main': ('s', '#ff7f0e'),      # Square, orange
            'learnable_gat': ('^', '#2ca02c'),       # Triangle up, green
            'learnable_sage': ('v', '#d62728'),      # Triangle down, red
            'learnable_no_step_emb': ('D', '#9467bd'), # Diamond, purple
            'learnable_no_edge_diff': ('*', '#8c564b'), # Star, brown
            'learnable_small_hidden': ('h', '#e377c2'), # Hexagon, pink
            'learnable_large_hidden': ('p', '#7f7f7f'), # Pentagon, gray
            'learnable_deep_gin': ('H', '#bcbd22'),   # Hexagon2, olive
            'learnable_shallow_gin': ('+', '#17becf'), # Plus, cyan
            'learnable_fixed_uniform': ('x', '#ff1744'), # X, deep red
            'learnable_fixed_cosine': ('|', '#00e676'),  # Vline, light green
            'learnable_dynamic_fw': ('_', '#3f51b5'),    # Hline, indigo
            'learnable_dynamic_ugd': ('1', '#ff5722'),   # Tri_down, deep orange
            'gradient_based': ('*', '#FF6B35'),          # Star, bright orange - 基于梯度的模型
        }
        
        # Define distinct markers for baseline models  
        baseline_markers = {
            'networkit_forest_fire': ('o', '#E74C3C'),      # Circle, red
            'networkit_local_degree': ('s', '#E67E22'),      # Square, orange  
            'networkit_local_similarity': ('^', '#F1C40F'),  # Triangle up, yellow
            'networkit_random_edge': ('v', '#2ECC71'),       # Triangle down, green
            'networkit_random_node_edge': ('D', '#3498DB'),  # Diamond, blue
            'networkit_scan': ('*', '#9B59B6'),              # Star, purple
            'networkit_simmelian': ('X', '#E91E63'),         # X, pink
            'pri_graphs': ('h', '#795548'),                  # Hexagon, brown
        }
        
        for result in results:
            model_name = result['model_name']
            category = result['model_info']['category']
            snr_auc = result['snr_auc']
            
            # Set colors and markers based on specific model
            if category == 'development':
                if model_name in development_markers:
                    marker, color = development_markers[model_name]
                    linestyle = '-'
                else:
                    # Fallback for unknown development models
                    color = '#1f77b4'  # Default blue
                    marker = 'o'
                    linestyle = '-'
                    
            elif category == 'baseline':
                if model_name in baseline_markers:
                    marker, color = baseline_markers[model_name]
                    linestyle = '--'
                else:
                    # Fallback for unknown baseline models
                    color = '#E74C3C'  # Default red
                    marker = 's'
                    linestyle = '--'
            else:
                # Unknown category
                color = 'green'
                marker = '^'
                linestyle = ':'
            
            plt.plot(
                result['complexity_metrics'], 
                result['information_metrics'],
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=8,
                alpha=0.8,  # Add transparency for better visualization
                label=f"{model_name} ({category}, AUC={snr_auc:.1f})"
            )
        
        plt.xlabel('Complexity Metric (L0 Norm)', fontsize=12)
        plt.ylabel('Information Metric', fontsize=12) 
        plt.title(f'Model Comparison - {dataset_name} + {downstream_model.upper()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_path = self.results_dir / f'model_comparison_{dataset_name}_{downstream_model}.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比图保存到: {plot_path}")
    
    def _plot_single_model(self, result: Dict, dataset_name: str, downstream_model: str):
        """为单个模型生成SNR曲线图"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        model_name = result['model_name']
        model_info = result['model_info']
        complexity_metrics = result['complexity_metrics']
        information_metrics = result['information_metrics']
        
        # 定义图标和颜色
        development_markers = {
            'learnable': ('o', '#1f77b4'),           # Circle, blue
            'learnable_main': ('s', '#ff7f0e'),      # Square, orange
            'learnable_gat': ('^', '#2ca02c'),       # Triangle up, green
            'learnable_sage': ('v', '#d62728'),      # Triangle down, red
            'learnable_no_step_emb': ('D', '#9467bd'), # Diamond, purple
            'learnable_no_edge_diff': ('*', '#8c564b'), # Star, brown
            'learnable_small_hidden': ('h', '#e377c2'), # Hexagon, pink
            'learnable_large_hidden': ('p', '#7f7f7f'), # Pentagon, gray
            'learnable_deep_gin': ('H', '#bcbd22'),   # Hexagon2, olive
            'learnable_shallow_gin': ('+', '#17becf'), # Plus, cyan
            'learnable_fixed_uniform': ('x', '#ff1744'), # X, deep red
            'learnable_fixed_cosine': ('|', '#00e676'),  # Vline, light green
            'learnable_dynamic_fw': ('_', '#3f51b5'),    # Hline, indigo
            'learnable_dynamic_ugd': ('1', '#ff5722'),   # Tri_down, deep orange
            'gradient_based': ('*', '#FF6B35'),          # Star, bright orange - 基于梯度的模型
        }
        
        baseline_markers = {
            'networkit_forest_fire': ('o', '#E74C3C'),      # Circle, red
            'networkit_local_degree': ('s', '#E67E22'),      # Square, orange  
            'networkit_local_similarity': ('^', '#F1C40F'),  # Triangle up, yellow
            'networkit_random_edge': ('v', '#2ECC71'),       # Triangle down, green
            'networkit_random_node_edge': ('D', '#3498DB'),  # Diamond, blue
            'networkit_scan': ('*', '#9B59B6'),              # Star, purple
            'networkit_simmelian': ('X', '#E91E63'),         # X, pink
            'pri_graphs': ('h', '#795548'),                  # Hexagon, brown
        }
        
        # 获取模型的图标和颜色
        if model_info['category'] == 'development':
            if model_name in development_markers:
                marker, color = development_markers[model_name]
            else:
                marker, color = 'o', '#1f77b4'  # Default: circle, blue
        elif model_info['category'] == 'baseline':
            if model_name in baseline_markers:
                marker, color = baseline_markers[model_name]
            else:
                marker, color = 's', '#E74C3C'  # Default: square, red
        else:
            marker, color = '^', 'green'  # Default: triangle, green
        
        # 绘制SNR曲线
        plt.plot(
            complexity_metrics, 
            information_metrics, 
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{model_name} (SNR-AUC: {result['snr_auc']:.2f})",
            linestyle='--' if model_info['category'] == 'baseline' else '-'
        )
        
        plt.xlabel('Complexity Metric (Edge Count)', fontsize=12)
        plt.ylabel('Information Metric', fontsize=12)
        plt.title(f'SNR Curve - {model_name} on {dataset_name} + {downstream_model.upper()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # 反转x轴，因为复杂度从高到低
        plt.gca().invert_xaxis()
        
        # 保存图表
        plot_path = self.results_dir / f'single_model_{model_name}_{dataset_name}_{downstream_model}.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 单模型SNR曲线图保存到: {plot_path}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """列出所有可用的模型"""
        dev_models = model_registry.list_development_models()
        baseline_models = model_registry.list_baseline_models()
        
        print(f"\n可用的Graph Summarization模型:")
        print(f"  开发模型 ({len(dev_models)} 个): {', '.join(dev_models)}")
        print(f"  基准模型 ({len(baseline_models)} 个): {', '.join(baseline_models)}")
        
        return {
            'development': dev_models,
            'baseline': baseline_models
        }

    def run_multi_task_benchmark(self,
                                 model_names: List[str],
                                 dataset_names: List[str] = None,
                                 task_types: List[str] = None,
                                 downstream_model: str = 'gcn',
                                 num_steps: int = 10,
                                 epochs: int = 100) -> Dict[str, Any]:
        """
        运行多任务基准测试，对多个数据集和任务类型进行测试

        Args:
            model_names: 要测试的模型名称列表
            dataset_names: 数据集名称列表，默认为所有支持的数据集
            task_types: 任务类型列表，默认为 ['original', 'degree']
            downstream_model: 下游任务模型类型
            num_steps: 图总结步数
            epochs: 训练轮数

        Returns:
            包含所有测试结果的字典
        """
        if dataset_names is None:
            # 排除太大的数据集
            dataset_names = ['Cora', 'CiteSeer', 'PubMed', 'KarateClub']

        if task_types is None:
            task_types = ['original', 'degree']

        print(f"\n{'='*120}")
        print(f"多任务基准测试")
        print(f"模型: {model_names}")
        print(f"数据集: {dataset_names}")
        print(f"任务类型: {task_types}")
        print(f"{'='*120}")

        all_results = {}
        summary_data = []

        total_experiments = len(model_names) * len(dataset_names) * len(task_types)
        current_experiment = 0

        for dataset_name in dataset_names:
            for task_type in task_types:
                experiment_key = f"{dataset_name}_{task_type}"
                print(f"\n{'='*80}")
                print(f"实验设置: {dataset_name} + {task_type} 任务")
                print(f"{'='*80}")

                experiment_results = []
                for model_name in model_names:
                    current_experiment += 1
                    print(f"\n[{current_experiment}/{total_experiments}] 测试 {model_name}...")

                    try:
                        result = self.run_single_model(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            task_type=task_type,
                            downstream_model=downstream_model,
                            num_steps=num_steps,
                            epochs=epochs
                        )

                        if result.get('success', False):
                            experiment_results.append(result)
                            summary_data.append({
                                'dataset': dataset_name,
                                'task': task_type,
                                'model': model_name,
                                'category': result['model_info']['category'],
                                'snr_auc': result['snr_auc'],
                                'training_time': result.get('training_time', 0),
                                'summarization_time': result.get('summarization_time', 0)
                            })
                        else:
                            print(f"❌ {model_name} 测试失败")

                    except Exception as e:
                        print(f"❌ {model_name} 测试出错: {e}")

                # 保存当前实验设置的结果
                if experiment_results:
                    all_results[experiment_key] = {
                        'results': experiment_results,
                        'dataset': dataset_name,
                        'task_type': task_type
                    }

                    # 生成当前实验的对比图
                    self._plot_comparison(experiment_results, f"{dataset_name}_{task_type}", downstream_model)

        # 生成综合报告
        self._generate_multi_task_report(summary_data, downstream_model)

        return {
            'all_results': all_results,
            'summary': summary_data,
            'total_experiments': total_experiments,
            'success_rate': len(summary_data) / total_experiments if total_experiments > 0 else 0
        }

    def _generate_multi_task_report(self, summary_data: List[Dict], downstream_model: str):
        """
        生成多任务测试的综合报告
        """
        if not summary_data:
            print("没有成功的测试结果，无法生成报告")
            return

        df = pd.DataFrame(summary_data)

        # 保存详细结果
        detailed_path = self.results_dir / f'multi_task_detailed_{downstream_model}.csv'
        df.to_csv(detailed_path, index=False)
        print(f"\n📊 详细多任务结果保存到: {detailed_path}")

        # 生成汇总表格
        pivot_table = df.pivot_table(
            values='snr_auc',
            index=['model', 'category'],
            columns=['dataset', 'task'],
            aggfunc='mean'
        )

        summary_path = self.results_dir / f'multi_task_summary_{downstream_model}.tsv'
        pivot_table.to_csv(summary_path, sep='\t')
        print(f"📊 多任务汇总表保存到: {summary_path}")

        # 打印汇总统计
        print(f"\n{'='*120}")
        print(f"多任务基准测试汇总 ({downstream_model.upper()})")
        print(f"{'='*120}")

        # 按模型分组统计
        model_stats = df.groupby(['model', 'category']).agg({
            'snr_auc': ['mean', 'std', 'count'],
            'training_time': 'mean',
            'summarization_time': 'mean'
        }).round(4)

        print("\n模型性能统计:")
        print(model_stats)

        # 最佳模型统计
        best_by_task = df.loc[df.groupby(['dataset', 'task'])['snr_auc'].idxmax()]
        print("\n各任务最佳模型:")
        for _, row in best_by_task.iterrows():
            print(f"  {row['dataset']} + {row['task']}: {row['model']} (SNR-AUC: {row['snr_auc']:.4f})")

    def compute_comprehensive_results_from_process_data(self,
                                                       experiment_dir: str,
                                                       model_names: List[str] = None) -> Dict[str, Any]:
        """
        单独的综合结果计算接口：根据已保存的过程结果计算综合结果

        这个方法符合DEMAND.md的要求，用于在所有模型的过程结果都完成后，
        单独计算综合结果（SNR曲线对比图和SNR-AUC表格）

        Args:
            experiment_dir: 实验目录路径 (如 "results/comprehensive_benchmark/Cora_original_gcn")
            model_names: 要包含的模型名称列表，如果为None则包含所有找到的模型

        Returns:
            综合结果字典
        """
        exp_dir = Path(experiment_dir)

        if not exp_dir.exists():
            raise ValueError(f"实验目录不存在: {experiment_dir}")

        process_dir = exp_dir / "process_results"
        if not process_dir.exists():
            raise ValueError(f"过程结果目录不存在: {process_dir}")

        print(f"🔄 从过程结果计算综合结果: {experiment_dir}")

        # 扫描所有可用的过程结果文件
        available_models = set()
        for file in process_dir.glob("*_step_metrics.tsv"):
            model_name = file.stem.replace("_step_metrics", "")
            available_models.add(model_name)

        if not available_models:
            raise ValueError(f"在 {process_dir} 中未找到任何过程结果文件")

        # 过滤模型
        if model_names is not None:
            available_models = available_models.intersection(set(model_names))
            missing_models = set(model_names) - available_models
            if missing_models:
                print(f"⚠️ 以下模型的过程结果未找到: {missing_models}")

        if not available_models:
            raise ValueError("没有可用的模型过程结果")

        print(f"📊 找到 {len(available_models)} 个模型的过程结果: {list(available_models)}")

        # 从过程结果文件加载数据
        results = []
        for model_name in available_models:
            try:
                # 读取步骤度量数据
                step_metrics_file = process_dir / f"{model_name}_step_metrics.tsv"
                df = pd.read_csv(step_metrics_file, sep='\t')

                complexity_metrics = df['complexity_metric'].tolist()

                # 检查是否有新的双重归一化指标
                if 'information_metric_log_ratio' in df.columns and 'information_metric_additive' in df.columns:
                    # 新格式：有双重归一化
                    information_metrics_log = df['information_metric_log_ratio'].tolist()
                    information_metrics_add = df['information_metric_additive'].tolist()
                    information_metrics = information_metrics_add  # 兼容旧字段，默认使用加法归一化
                else:
                    # 旧格式：只有单一归一化（假设为加法归一化）
                    information_metrics = df['information_metric'].tolist()
                    information_metrics_add = information_metrics
                    information_metrics_log = information_metrics  # 用相同值填充

                # 计算IC-AUC（两种归一化）
                from ..metrics import ICAnalysis
                ic_auc_log = ICAnalysis.compute_ic_auc(complexity_metrics, information_metrics_log)
                ic_auc_add = ICAnalysis.compute_ic_auc(complexity_metrics, information_metrics_add)

                # 计算信息阈值点
                threshold_point_log = ICAnalysis.compute_information_threshold_point(
                    complexity_metrics, information_metrics_log, threshold=0.8)
                threshold_point_add = ICAnalysis.compute_information_threshold_point(
                    complexity_metrics, information_metrics_add, threshold=0.8)

                # 向后兼容
                snr_auc = ic_auc_add

                # 尝试获取模型信息（如果可用）
                model_category = 'unknown'
                model_description = f"Model {model_name}"

                # 基于模型名称推断类别
                if 'networkit' in model_name or 'pri_graphs' in model_name:
                    model_category = 'baseline'
                elif 'neural_enhanced' in model_name or 'gradient_based' in model_name:
                    model_category = 'development'

                result = {
                    'model_name': model_name,
                    'model_info': {
                        'category': model_category,
                        'description': model_description
                    },
                    'complexity_metrics': complexity_metrics,
                    # 双重归一化信息度量
                    'information_metrics_log_ratio': information_metrics_log,
                    'information_metrics_additive': information_metrics_add,
                    # IC-AUC指标（两种归一化）
                    'ic_auc_log_ratio': ic_auc_log,
                    'ic_auc_additive': ic_auc_add,
                    # 信息阈值点（两种归一化）
                    'threshold_point_log_ratio': threshold_point_log,
                    'threshold_point_additive': threshold_point_add,
                    # 向后兼容的字段
                    'information_metrics': information_metrics,
                    'snr_auc': snr_auc,
                    # 默认值（从过程数据重建时不可用）
                    'training_time': 0.0,
                    'summarization_time': 0.0,
                    'training_history': None,
                    'summary_graphs': None,
                    'original_graph': None,
                    'success': True,
                    'loaded_from_process_data': True
                }

                results.append(result)
                print(f"  ✅ {model_name}: IC-AUC(log)={ic_auc_log:.4f}, IC-AUC(add)={ic_auc_add:.4f}")

            except Exception as e:
                print(f"  ❌ 无法加载 {model_name} 的过程结果: {e}")
                continue

        if not results:
            raise ValueError("无法加载任何有效的过程结果")

        # 创建comprehensive_results目录
        comprehensive_dir = exp_dir / "comprehensive_results"
        comprehensive_dir.mkdir(exist_ok=True)

        # 解析实验信息（从目录名）
        exp_parts = exp_dir.name.split('_')
        if len(exp_parts) >= 3:
            dataset_name = exp_parts[0]
            task_type = exp_parts[1]
            downstream_model = exp_parts[2]
        else:
            dataset_name = "Unknown"
            task_type = "unknown"
            downstream_model = "unknown"

        # 为结果添加实验信息
        for result in results:
            result['dataset'] = dataset_name
            result['task_type'] = task_type
            result['downstream_model'] = downstream_model
            result['exp_dir'] = exp_dir
            result['exp_id'] = exp_dir.name

        # 生成综合结果
        self._save_comprehensive_results(results, exp_dir)

        # 生成详细对比报告（可选功能，忽略错误）
        try:
            if hasattr(self, '_generate_detailed_comparison'):
                self._generate_detailed_comparison(results, dataset_name, downstream_model)
            if hasattr(self, '_generate_comparison_report'):
                self._generate_comparison_report(results, dataset_name, downstream_model)
            if hasattr(self, '_plot_comparison'):
                self._plot_comparison(results, dataset_name, downstream_model)
        except Exception as e:
            print(f"⚠️ 可选的详细报告生成跳过: {e}")

        print(f"✅ 综合结果计算完成，保存到: {comprehensive_dir}")

        return {
            'experiment_dir': str(exp_dir),
            'models_processed': list(available_models),
            'comprehensive_results_dir': str(comprehensive_dir),
            'results': results
        }

    def batch_compute_comprehensive_results(self,
                                          results_base_dir: str = None,
                                          experiment_pattern: str = "*",
                                          model_names: List[str] = None) -> Dict[str, Any]:
        """
        批量计算多个实验的综合结果

        Args:
            results_base_dir: 结果基础目录，默认使用self.results_dir
            experiment_pattern: 实验目录匹配模式（如 "Cora_*_gcn"）
            model_names: 要包含的模型名称列表

        Returns:
            批量处理结果字典
        """
        if results_base_dir is None:
            results_base_dir = self.results_dir
        else:
            results_base_dir = Path(results_base_dir)

        print(f"🔄 批量计算综合结果: {results_base_dir} / {experiment_pattern}")

        # 找到所有匹配的实验目录
        experiment_dirs = list(results_base_dir.glob(experiment_pattern))
        experiment_dirs = [d for d in experiment_dirs if d.is_dir()]

        if not experiment_dirs:
            raise ValueError(f"未找到匹配的实验目录: {results_base_dir} / {experiment_pattern}")

        print(f"📂 找到 {len(experiment_dirs)} 个实验目录")

        batch_results = {}
        successful_count = 0

        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            print(f"\n🔬 处理实验: {exp_name}")

            try:
                result = self.compute_comprehensive_results_from_process_data(
                    experiment_dir=str(exp_dir),
                    model_names=model_names
                )
                batch_results[exp_name] = result
                successful_count += 1

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                batch_results[exp_name] = {'error': str(e)}

        print(f"\n✅ 批量处理完成: {successful_count}/{len(experiment_dirs)} 个实验成功")

        return {
            'processed_experiments': batch_results,
            'success_count': successful_count,
            'total_count': len(experiment_dirs)
        }

    def _log_memory_usage(self, stage: str = ""):
        """Log current memory usage"""
        if not self.memory_monitor:
            return

        # System memory
        system_memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()

        print(f"🔍 Memory usage{f' ({stage})' if stage else ''}:")
        print(f"  System: {system_memory.used / 1024**3:.2f}GB / {system_memory.total / 1024**3:.2f}GB "
              f"({system_memory.percent:.1f}%)")
        print(f"  Process: {process_memory.rss / 1024**3:.2f}GB")

        # CUDA memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  CUDA {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached / {total:.2f}GB total")

    def _cleanup_memory(self):
        """Perform comprehensive memory cleanup"""
        # Collect garbage
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force synchronization
            torch.cuda.synchronize()

        print("🧹 Memory cleanup completed")

    def _get_adaptive_epochs(self, requested_epochs: int, num_steps: int) -> int:
        """Calculate adaptive epochs based on available memory and number of steps"""
        if not self.enable_memory_optimization:
            return requested_epochs

        # Basic heuristic: reduce epochs for larger step counts
        if num_steps > 15:
            return max(requested_epochs // 3, 20)
        elif num_steps > 10:
            return max(requested_epochs // 2, 30)
        else:
            return requested_epochs

    def _estimate_memory_requirements(self, graph: Data, num_steps: int) -> float:
        """Estimate memory requirements in GB for a given graph and step count"""
        num_nodes = graph.num_nodes
        num_edges = graph.edge_index.shape[1]
        feature_dim = graph.x.shape[1]

        # Rough estimation based on graph size
        graph_memory_mb = (num_nodes * feature_dim * 4 + num_edges * 2 * 4) / 1024**2  # MB
        model_memory_mb = 50  # Typical downstream model size in MB
        total_memory_gb = (graph_memory_mb + model_memory_mb * num_steps) / 1024

        return total_memory_gb

    def _check_memory_feasibility(self, graph: Data, num_steps: int) -> Tuple[bool, str]:
        """Check if the computation is feasible with current memory"""
        estimated_memory = self._estimate_memory_requirements(graph, num_steps)

        if torch.cuda.is_available():
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_memory = device_memory - torch.cuda.memory_allocated(0) / 1024**3

            if estimated_memory > free_memory * 0.8:  # Use 80% threshold
                return False, f"Estimated memory ({estimated_memory:.2f}GB) exceeds available ({free_memory:.2f}GB)"

        return True, "Memory check passed"

    def run_ppi_multi_label_benchmark(self,
                                     model_names: List[str],
                                     label_indices: List[int] = None,
                                     downstream_model: str = 'gcn',
                                     num_steps: int = 5,
                                     epochs: int = 30) -> Dict[str, Any]:
        """
        Run benchmark on PPI dataset for multiple binary classification tasks

        Args:
            model_names: List of model names to test
            label_indices: List of label indices to test (0-120). If None, test first 10 labels
            downstream_model: Downstream model type
            num_steps: Number of summarization steps
            epochs: Training epochs

        Returns:
            Dictionary containing results for all label tasks
        """
        if label_indices is None:
            # Test first 10 labels by default to avoid excessive computation
            label_indices = list(range(10))

        print(f"\n{'='*120}")
        print(f"PPI Multi-Label Benchmark")
        print(f"模型: {model_names}")
        print(f"标签任务: {len(label_indices)} 个 (indices: {label_indices})")
        print(f"{'='*120}")

        all_results = {}
        summary_data = []

        total_experiments = len(model_names) * len(label_indices)
        current_experiment = 0

        for label_idx in label_indices:
            task_key = f"PPI_label_{label_idx}"
            print(f"\n{'='*80}")
            print(f"PPI 标签任务 {label_idx}")
            print(f"{'='*80}")

            task_results = []
            for model_name in model_names:
                current_experiment += 1
                print(f"\n[{current_experiment}/{total_experiments}] 测试 {model_name} on label {label_idx}...")

                try:
                    # Include PPI label index in model kwargs
                    model_kwargs = {'ppi_label_index': label_idx}

                    result = self.run_single_model(
                        model_name=model_name,
                        dataset_name='PPI',
                        task_type='original',
                        downstream_model=downstream_model,
                        num_steps=num_steps,
                        epochs=epochs,
                        model_kwargs=model_kwargs
                    )

                    if result.get('success', False):
                        task_results.append(result)
                        summary_data.append({
                            'label_index': label_idx,
                            'model': model_name,
                            'category': result['model_info']['category'],
                            'snr_auc': result['snr_auc'],
                            'training_time': result.get('training_time', 0),
                            'summarization_time': result.get('summarization_time', 0)
                        })
                        print(f"✅ {model_name} on label {label_idx}: SNR-AUC = {result['snr_auc']:.4f}")
                    else:
                        print(f"❌ {model_name} on label {label_idx} 测试失败")

                except Exception as e:
                    print(f"❌ {model_name} on label {label_idx} 测试出错: {e}")

            # Save results for this label task
            if task_results:
                all_results[task_key] = {
                    'results': task_results,
                    'label_index': label_idx,
                    'dataset': 'PPI'
                }

        # Generate comprehensive report
        self._generate_ppi_multi_label_report(summary_data, downstream_model)

        return {
            'all_results': all_results,
            'summary': summary_data,
            'total_experiments': total_experiments,
            'success_rate': len(summary_data) / total_experiments if total_experiments > 0 else 0,
            'tested_labels': label_indices
        }

    def _generate_ppi_multi_label_report(self, summary_data: List[Dict], downstream_model: str):
        """
        Generate comprehensive report for PPI multi-label benchmark
        """
        if not summary_data:
            print("没有成功的测试结果，无法生成报告")
            return

        df = pd.DataFrame(summary_data)

        # Save detailed results
        detailed_path = self.results_dir / f'ppi_multi_label_detailed_{downstream_model}.csv'
        df.to_csv(detailed_path, index=False)
        print(f"\n📊 PPI多标签详细结果保存到: {detailed_path}")

        # Generate summary table by model
        model_summary = df.groupby(['model', 'category']).agg({
            'snr_auc': ['mean', 'std', 'min', 'max', 'count'],
            'training_time': 'mean',
            'summarization_time': 'mean'
        }).round(4)

        summary_path = self.results_dir / f'ppi_multi_label_summary_{downstream_model}.tsv'
        model_summary.to_csv(summary_path, sep='\t')
        print(f"📊 PPI多标签汇总表保存到: {summary_path}")

        # Print summary statistics
        print(f"\n{'='*120}")
        print(f"PPI多标签基准测试汇总 ({downstream_model.upper()})")
        print(f"{'='*120}")

        print(f"测试的标签数量: {df['label_index'].nunique()}")
        print(f"测试的模型数量: {df['model'].nunique()}")
        print(f"总实验次数: {len(df)}")

        print(f"\n模型平均性能 (跨所有标签):")
        for (model, category), group in df.groupby(['model', 'category']):
            mean_auc = group['snr_auc'].mean()
            std_auc = group['snr_auc'].std()
            print(f"  {model} ({category}): {mean_auc:.4f} ± {std_auc:.4f}")

        # Best performing label tasks
        best_by_label = df.loc[df.groupby('label_index')['snr_auc'].idxmax()]
        print(f"\n各标签任务最佳模型:")
        for _, row in best_by_label.iterrows():
            print(f"  Label {row['label_index']}: {row['model']} (SNR-AUC: {row['snr_auc']:.4f})")