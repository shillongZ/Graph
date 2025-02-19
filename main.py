# 导入需要的包，遇到安装问题可在官方文档或其他文章查找解决方案
import torch
import torch.nn.functional as F
import numpy as np
# 导入GCN层、GraphSAGE层和GAT层
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid
from util_functions import load_data_set, use_cuda, get_data_split, symmetric_normalize_adj
from torch_geometric.data import Data
# 加载数据，出错可自行下载，解决方案见下文

def train(args):
    device = use_cuda()

    [c_train, c_val] = args.train_val_class
    idx, labellist, edge, features = load_data_set(args.dataset)
    idx_train, idx_test = get_data_split(idx=idx, labellist=labellist)
    y_true = np.array([int(temp[0]) for temp in labellist]) #[n, 1]
    num_class = len(set(y_true))
    y_true = torch.from_numpy(y_true).type(torch.LongTensor).to(device)
    edge_index = torch.tensor(edge, dtype=torch.long)
    data = Data(x=features, edge_index=edge_index).to(device)

    class GCN_NET(torch.nn.Module):

        def __init__(self, features, hidden, classes):
            super(GCN_NET, self).__init__()
            self.conv1 = GCNConv(features, hidden)  # shape（输入的节点特征维度 * 中间隐藏层的维度）
            self.conv2 = GCNConv(hidden, classes)  # shaape（中间隐藏层的维度 * 节点类别）

        def forward(self, data):
            # 加载节点特征和邻接关系
            x, edge_index = data.x, data.edge_index
            # 传入卷积层
            x = self.conv1(x, edge_index)
            x = F.relu(x)  # 激活函数
            x = F.dropout(x, training=self.training)  # dropout层，防止过拟合
            x = self.conv2(x, edge_index)  # 第二层卷积层
            # 将经过两层卷积得到的特征输入log_softmax函数得到概率分布
            return F.log_softmax(x, dim=1)


    # 判断是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建模型，设置中间隐藏层维度为16
    model = GCN_NET(features.shape[1], 64, num_class).to(device)
    # 定义优化函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(600):
        optimizer.zero_grad() # 梯度设为零
        out = model(data)  # 模型输出
        loss = F.nll_loss(out[idx_train], y_true[idx_train])  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 一步优化


    model.eval()  # 评估模型
    _, pred = model(data).max(dim=1)  # 得到模型输出的类别
    correct = int(pred[idx_test].eq(y_true[idx_test]).sum().item())  # 计算正确的个数
    acc = correct / int(len(idx_test))  # 得出准确率
    print('GCN Accuracy: {:.4f}'.format(acc))
    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='C-M10-M', choices=['cora', 'citeseer', 'C-M10-M','small_matrix','big_matrix'], help="dataset")
    parser.add_argument("--train-val-class", type=int, nargs='*', default=[3, 0], help="the first #train_class and #validation classes")
    args = parser.parse_args()
    print(args)
    test_acc = []
    for i in range(10):
        cur_acc = train(args)
        test_acc.append(cur_acc)
    print("{:.4f}".format(np.mean(test_acc)), "{:.4f}".format(np.var(test_acc)))