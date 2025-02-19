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
split_idx = dataset.get_idx_split()

train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

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
model = GCN(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 定义训练函数
def train():
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx].squeeze(1))
    loss.backward()
    optimizer.step()

    return loss.item()

# 定义评估函数
def test():
    model.eval()

    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=1, keepdim=True)

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': data.y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[valid_idx],
        'y_pred': y_pred[valid_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc

# 训练模型
for epoch in range(1, 1001):
    loss = train()
    train_acc, valid_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Val: {valid_acc:.4f}, Test: {test_acc:.4f}')