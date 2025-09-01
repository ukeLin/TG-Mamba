import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn.cluster import DBSCAN

class VPTree:
    """VP-Tree implementation for fast nearest neighbor search"""
    def __init__(self, points, distance_fn):
        self.points = points
        self.distance_fn = distance_fn
        self.left = None
        self.right = None
        self.vp = None
        self.median = None
        
        if len(points) > 0:
            # 随机选择制高点
            self.vp = points[np.random.randint(len(points))]
            distances = [distance_fn(self.vp, p) for p in points if not np.array_equal(p, self.vp)]
            
            if len(distances) > 0:
                self.median = np.median(distances)
                
                # 分割点集
                left_points = [p for p in points if not np.array_equal(p, self.vp) and 
                             distance_fn(self.vp, p) <= self.median]
                right_points = [p for p in points if not np.array_equal(p, self.vp) and 
                              distance_fn(self.vp, p) > self.median]
                
                if left_points:
                    self.left = VPTree(left_points, distance_fn)
                if right_points:
                    self.right = VPTree(right_points, distance_fn)
    
    def get_neighbors_within_radius(self, query_point, radius):
        """获取给定半径内的邻居"""
        neighbors = []
        self._get_neighbors_within_radius(query_point, radius, neighbors)
        return neighbors
    
    def _get_neighbors_within_radius(self, query_point, radius, neighbors):
        if self.vp is None:
            return
        
        d = self.distance_fn(query_point, self.vp)
        if d <= radius:
            neighbors.append(self.vp)
        
        if self.median is not None:
            if d - radius <= self.median and self.left:
                self.left._get_neighbors_within_radius(query_point, radius, neighbors)
            if d + radius > self.median and self.right:
                self.right._get_neighbors_within_radius(query_point, radius, neighbors)

class LocalTypicalityMILHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512, epsilon=5.0, k_instances=10):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.k_instances = k_instances
        
        # 特征转换层
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def compute_pairwise_distances(self, instances):
        """计算成对距离矩阵，使用分块计算减少内存使用"""
        n = len(instances)
        distances = torch.zeros((n, n), device=instances.device)
        
        # 使用分块计算，每块大小为1024
        block_size = 1024
        for i in range(0, n, block_size):
            end_i = min(i + block_size, n)
            for j in range(0, n, block_size):
                end_j = min(j + block_size, n)
                # 计算当前块的距离
                distances[i:end_i, j:end_j] = torch.cdist(
                    instances[i:end_i], instances[j:end_j], p=2
                )
        
        return distances

    def find_epsilon_neighbors(self, instances, distances=None):
        """找到每个实例的epsilon邻域"""
        n = len(instances)
        device = instances.device
        
        # 如果没有提供距离矩阵，则计算
        if distances is None:
            distances = self.compute_pairwise_distances(instances)
        
        # 批量找到所有邻居
        neighbors_mask = distances < self.epsilon
        return [torch.where(neighbors_mask[i])[0] for i in range(n)]

    def compute_local_typicality(self, instances, neighbors, distances=None):
        """计算局部典型度"""
        device = instances.device
        n = len(instances)
        
        # 如果没有提供距离矩阵，则计算
        if distances is None:
            distances = self.compute_pairwise_distances(instances)
        
        # 计算全局带宽参数h
        s = torch.std(distances)
        if s < 1e-6:
            s = torch.mean(distances) + 1e-6
        h = 1.06 * s * (n ** (-1/5))
        h = max(h, 1e-3)
        
        # 批量计算高斯核
        typicalities = torch.zeros(n, device=device)
        
        # 分块计算典型度，避免内存溢出
        block_size = 128
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            
            # 对当前块中的每个实例单独计算
            for i in range(start, end):
                if len(neighbors[i]) == 0:
                    typicalities[i] = 0.0
                    continue
                
                # 获取当前实例的邻居距离
                neighbor_distances = distances[i, neighbors[i]]
                
                # 计算高斯核
                gaussian_kernel = torch.exp(-neighbor_distances**2 / (2.0 * h * h))
                typicalities[i] = torch.mean(gaussian_kernel)
        
        return typicalities

    def select_key_instances(self, instances, typicalities, bag_label):
        """选择关键实例和对立实例
        Args:
            instances: 实例特征
            typicalities: 典型度值
            bag_label: 包的标签 (0: low, 1: high)
        """
        device = instances.device
        typicalities = typicalities.to(device)
        
        # 计算实例间的距离矩阵
        distances = self.compute_pairwise_distances(instances)
        
        # 根据包标签选择正关键实例（与包标签相同类别的典型实例）
        if bag_label == 1:  # high bag
            # 选择典型度最高的k个实例作为正关键实例（代表high类）
            _, top_indices = torch.topk(typicalities, min(self.k_instances, len(typicalities)))
            pos_indices = top_indices  # 这些是high的关键实例
        else:  # low bag
            # 选择典型度最低的k个实例作为正关键实例（代表low类）
            _, top_indices = torch.topk(-typicalities, min(self.k_instances, len(typicalities)))
            pos_indices = top_indices  # 这些是low的关键实例
        
        # 对于每个正关键实例，找到与它距离最远的实例作为负关键实例候选
        neg_candidates = []
        for idx in pos_indices:
            # 获取与当前实例的距离
            instance_distances = distances[idx].clone()
            # 排除已选的正关键实例
            instance_distances[pos_indices] = -1
            # 选择距离最远的实例
            _, furthest = torch.topk(instance_distances, min(self.k_instances, len(instance_distances)))
            neg_candidates.extend(furthest.tolist())
        
        # 从候选中选择最终的负关键实例
        # 对于high bag，负关键实例应该是典型度最低的
        # 对于low bag，负关键实例应该是典型度最高的
        neg_candidates = list(set(neg_candidates) - set(pos_indices.tolist()))
        if len(neg_candidates) > 0:
            neg_candidates = torch.tensor(neg_candidates, device=device)
            if bag_label == 1:  # high bag
                # 选择典型度最低的作为负关键实例（代表low类）
                neg_typicalities = -typicalities[neg_candidates]  # 取负使得最低的典型度得到最高的分数
            else:  # low bag
                # 选择典型度最高的作为负关键实例（代表high类）
                neg_typicalities = typicalities[neg_candidates]
            
            _, top_neg = torch.topk(neg_typicalities, min(self.k_instances, len(neg_typicalities)))
            neg_indices = neg_candidates[top_neg]
        else:
            # 如果没有合适的负候选，选择与正关键实例最远的实例
            mean_distances = torch.mean(distances[pos_indices], dim=0)
            mean_distances[pos_indices] = -1  # 排除正关键实例
            _, neg_indices = torch.topk(mean_distances, min(self.k_instances, len(mean_distances)))
        
        key_instances = {
            'positive': pos_indices,  # 与包标签相同类别的关键实例
            'negative': neg_indices   # 与包标签相反类别的关键实例
        }
        
        return key_instances, None

    def handle_noise_points(self, instances, noise_idx, key_instances, typicalities, epsilon_neighbors, distances=None):
        """处理噪声点的软标签分配"""
        device = instances.device
        neighbors = epsilon_neighbors[noise_idx]
        
        # 如果没有关键实例，返回中性标签
        if len(key_instances['positive']) == 0 and len(key_instances['negative']) == 0:
            return torch.tensor([0.5, 0.5], device=device)
        
        # 如果没有提供距离矩阵，则按需计算距离
        if distances is None:
            # 只计算当前噪声点与关键实例的距离
            noise_instance = instances[noise_idx:noise_idx+1]
            pos_distances = torch.cdist(noise_instance, instances[key_instances['positive']], p=2)[0]
            neg_distances = torch.cdist(noise_instance, instances[key_instances['negative']], p=2)[0]
        else:
            pos_distances = distances[noise_idx, key_instances['positive']]
            neg_distances = distances[noise_idx, key_instances['negative']]
        
        # 找出邻域内的关键实例
        pos_neighbors = torch.where(pos_distances < self.epsilon)[0]
        neg_neighbors = torch.where(neg_distances < self.epsilon)[0]
        
        # 如果邻域内有关键实例，使用典型性加权投票
        if len(pos_neighbors) + len(neg_neighbors) > 0:
            pos_weight = sum(typicalities[key_instances['positive'][i]] for i in pos_neighbors)
            neg_weight = sum(typicalities[key_instances['negative'][i]] for i in neg_neighbors)
            
            total_weight = pos_weight + neg_weight
            if total_weight > 0:
                pos_prob = pos_weight / total_weight
                return torch.tensor([1-pos_prob, pos_prob], device=device)
        
        # 如果邻域内没有关键实例，使用全局相似度
        pos_similarities = torch.exp(-pos_distances).mean() if len(key_instances['positive']) > 0 else 0
        neg_similarities = torch.exp(-neg_distances).mean() if len(key_instances['negative']) > 0 else 0
        
        # 归一化相似度得到软标签
        total = pos_similarities + neg_similarities + 1e-6
        soft_label = torch.tensor([
            neg_similarities / total,
            pos_similarities / total
        ], device=device)
        
        return soft_label

    def assign_soft_labels(self, instances, key_instances, typicalities, epsilon_neighbors, distances):
        """为所有实例分配软标签"""
        device = instances.device
        n = len(instances)
        soft_labels = torch.zeros((n, 2), device=device)
        
        # 为每个实例分配软标签
        for i in range(n):
            soft_labels[i] = self.handle_noise_points(
                instances, i, key_instances, typicalities, epsilon_neighbors, distances
            )
        
        return soft_labels

    def visualize_feature_space(self, instances, typicalities, key_instances, bag_label, soft_labels, save_path=None):
        """可视化特征空间和关键实例"""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
        
        # 将特征转换到2D空间
        tsne = TSNE(n_components=2, random_state=42)
        instances_2d = tsne.fit_transform(instances.detach().cpu().numpy())
        
        # 创建图形
        plt.figure(figsize=(20, 8))
        
        # 1. 可视化典型度分布
        plt.subplot(231)
        scatter = plt.scatter(instances_2d[:, 0], instances_2d[:, 1], 
                            c=typicalities.detach().cpu().numpy(),
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Typicality')
        plt.title(f'Typicality Distribution (Bag Label: {bag_label})')
        
        # 2. 可视化关键实例
        plt.subplot(232)
        # 绘制所有实例（灰色背景）
        plt.scatter(instances_2d[:, 0], instances_2d[:, 1], 
                   color='lightgray', alpha=0.3, label='All Instances')
        
        # 绘制正关键实例
        if len(key_instances['positive']) > 0:
            pos_indices = key_instances['positive'].cpu().numpy()
            plt.scatter(instances_2d[pos_indices, 0], instances_2d[pos_indices, 1],
                       color='red', alpha=0.8, label='Positive Key')
        
        # 绘制负关键实例
        if len(key_instances['negative']) > 0:
            neg_indices = key_instances['negative'].cpu().numpy()
            plt.scatter(instances_2d[neg_indices, 0], instances_2d[neg_indices, 1],
                       color='blue', alpha=0.8, label='Negative Key')
        
        plt.legend()
        plt.title('Key Instances')
        
        # 3. 可视化硬标签（从包标签继承）
        plt.subplot(233)
        hard_labels = np.full(len(instances), bag_label)
        colors = ['blue', 'red']  # 0: blue (negative), 1: red (positive)
        for label in [0, 1]:
            mask = hard_labels == label
            plt.scatter(instances_2d[mask, 0], instances_2d[mask, 1], 
                       color=colors[label], alpha=0.6,
                       label=f'Class {label}')
        plt.legend()
        plt.title('Hard Labels (Inherited from Bag)')
        
        # 4. 可视化软标签分布（二分类形式）
        plt.subplot(234)
        soft_probs = soft_labels[:, 1].detach().cpu().numpy()  # 正类的概率
        soft_binary = (soft_probs > 0.5).astype(int)  # 转换为二分类
        colors = ['blue', 'red']  # 0: blue (negative), 1: red (positive)
        for label in [0, 1]:
            mask = soft_binary == label
            plt.scatter(instances_2d[mask, 0], instances_2d[mask, 1], 
                       color=colors[label], alpha=0.6,
                       label=f'Class {label} (prob {"<" if label==0 else ">"} 0.5)')
        plt.legend()
        plt.title('Soft Labels (Binary)')
        
        # 5. 可视化实例关系
        plt.subplot(235)
        # 绘制所有实例
        plt.scatter(instances_2d[:, 0], instances_2d[:, 1], 
                   color='lightgray', alpha=0.3)
        
        # 绘制正关键实例和它们之间的连接
        if len(key_instances['positive']) > 0:
            pos_indices = key_instances['positive'].cpu().numpy()
            plt.scatter(instances_2d[pos_indices, 0], instances_2d[pos_indices, 1],
                       color='red', alpha=0.8, label='Positive Key')
            # 绘制正关键实例之间的连接
            for i in range(len(pos_indices)):
                for j in range(i+1, len(pos_indices)):
                    plt.plot([instances_2d[pos_indices[i], 0], instances_2d[pos_indices[j], 0]],
                            [instances_2d[pos_indices[i], 1], instances_2d[pos_indices[j], 1]],
                            'r-', alpha=0.2)
        
        # 绘制负关键实例和它们与正关键实例的连接
        if len(key_instances['negative']) > 0:
            neg_indices = key_instances['negative'].cpu().numpy()
            plt.scatter(instances_2d[neg_indices, 0], instances_2d[neg_indices, 1],
                       color='blue', alpha=0.8, label='Negative Key')
            # 绘制与最近的正关键实例的连接
            for neg_idx in neg_indices:
                if len(pos_indices) > 0:
                    # 找到最近的正关键实例
                    distances = np.sqrt(np.sum((instances_2d[pos_indices] - instances_2d[neg_idx])**2, axis=1))
                    nearest_pos = pos_indices[np.argmin(distances)]
                    plt.plot([instances_2d[neg_idx, 0], instances_2d[nearest_pos, 0]],
                            [instances_2d[neg_idx, 1], instances_2d[nearest_pos, 1]],
                            'b--', alpha=0.2)
        
        plt.legend()
        plt.title('Instance Relationships')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def forward(self, x, labels):
        """前向传播"""
        device = x.device
        batch_size = x.shape[0]
        
        # 特征提取
        x = self.feature_extractor(x)
        
        # 对每个包单独处理
        all_instance_logits = []
        all_typicalities = []
        all_soft_labels = []
        all_bag_predictions = []
        
        # 初始化或更新batch计数器
        if not hasattr(self, 'batch_count'):
            self.batch_count = 0
        self.batch_count += 1
        
        for i in range(batch_size):
            # 获取当前包的实例
            instances = x[i]  # [N, hidden_dim]
            
            # 计算一次距离矩阵，后续复用
            distances = self.compute_pairwise_distances(instances)
            
            # 找到epsilon邻域
            epsilon_neighbors = self.find_epsilon_neighbors(instances, distances)
            
            # 计算典型度
            typicalities = self.compute_local_typicality(instances, epsilon_neighbors, distances)
            
            # 选择关键实例
            key_instances, _ = self.select_key_instances(instances, typicalities, labels[i])
            
            # 分配软标签
            soft_labels = self.assign_soft_labels(
                instances, key_instances, typicalities, epsilon_neighbors, distances
            )
            
            # 每5个batch保存第一个包的可视化结果
            if self.batch_count % 5 == 0 and i == 0:
                save_path = f'feature_space_batch{self.batch_count}_bag{i}.png'
                self.visualize_feature_space(
                    instances, typicalities, key_instances, 
                    labels[i].item(), soft_labels, save_path
                )
            
            # 计算实例级别的预测
            instance_logits = self.classifier(instances)
            
            # 使用典型度加权聚合得到包级别的预测
            weighted_predictions = soft_labels * typicalities.unsqueeze(-1)  # [N, 2]
            bag_predictions = weighted_predictions.sum(dim=0)  # [2]
            bag_predictions = bag_predictions / (typicalities.sum() + 1e-6)  # 归一化
            
            # 收集结果
            all_instance_logits.append(instance_logits)
            all_typicalities.append(typicalities)
            all_soft_labels.append(soft_labels)
            all_bag_predictions.append(bag_predictions)
        
        # 堆叠所有结果
        all_instance_logits = torch.stack(all_instance_logits)
        all_typicalities = torch.stack(all_typicalities)
        all_soft_labels = torch.stack(all_soft_labels)
        all_bag_predictions = torch.stack(all_bag_predictions)
        
        return all_instance_logits, all_typicalities, all_soft_labels, all_bag_predictions 