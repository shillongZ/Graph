import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# 设定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据集
dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=None)
split_idx = dataset.get_idx_split()
data = dataset[0]

# 数据集转移到设备
data = data.to(device)

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

# 实例化模型和优化器
model = GCN(hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()