import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
import os
from dataset import CGRDataset

class Model_SAGE(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        # 实例化SAGEConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=16, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=16, out_feats=32, aggregator_type='mean')
        self.conv3 = dglnn.SAGEConv(
            in_feats=32, out_feats=64, aggregator_type='mean')
        self.conv4 = dglnn.SAGEConv(
            in_feats=64, out_feats=128, aggregator_type='mean')
        self.conv5 = dglnn.SAGEConv(
            in_feats=128, out_feats=128, aggregator_type='mean')
        self.conv6 = dglnn.SAGEConv(
            in_feats=128, out_feats=64, aggregator_type='mean')
        self.conv7 = dglnn.SAGEConv(
            in_feats=64, out_feats=32, aggregator_type='mean')
        self.conv8 = dglnn.SAGEConv(
            in_feats=32, out_feats=16, aggregator_type='mean')

        self.res1=nn.Linear(in_feats,32)
        self.res2=nn.Linear(32,128)
        self.res3 = nn.Linear(128, 64)
        self.res4 = nn.Linear(64, 16)

        self.lin1=nn.Linear(16,128)
        self.lin2 = nn.Linear(128, out_feats)

        self.bn1=torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.bn6 = torch.nn.BatchNorm1d(64)
        self.bn7 = torch.nn.BatchNorm1d(32)
        self.bn8 = torch.nn.BatchNorm1d(16)

        #参数初始化
        torch.nn.init.kaiming_uniform_(self.conv1.fc_neigh.weight,mode='fan_in',nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.fc_neigh.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv3.fc_neigh.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv4.fc_neigh.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv5.fc_neigh.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.res1.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.res2.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.lin1.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.lin2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h=self.bn1(h)
        h = F.leaky_relu(h)
        h=F.dropout(h,p=0.2,training=self.training)
        h = self.conv2(graph, h)
        h = h + self.res1(inputs)
        h=self.bn2(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h1=self.conv3(graph,h)
        h1=self.bn3(h1)
        h1 = F.leaky_relu(h1)
        h1 = self.conv4(graph, h1)
        h1 = h1 + self.res2(h)
        h1=self.bn4(h1)
        h1 = F.leaky_relu(h1)

        h2 = self.conv5(graph, h1)
        h2=self.bn5(h2)
        h2 = F.leaky_relu(h2)
        h2 = self.conv6(graph, h2)
        h2=h2+self.res3(h1)
        h2=self.bn6(h2)
        h2 = F.leaky_relu(h2)

        h3 = self.conv7(graph, h2)
        h3=self.bn7(h3)
        h3 = F.leaky_relu(h3)
        #h = F.dropout(h, p=0.2, training=self.training)
        h3 = self.conv8(graph, h3)
        h3 = h3 + self.res4(h2)
        h3=self.bn8(h3)
        h3 = F.leaky_relu(h3)
        #h3 = F.dropout(h3, p=0.2, training=self.training)

        h3=self.lin1(h3)
        h3=F.leaky_relu(h3)
        #h3 = F.dropout(h3, p=0.2, training=self.training)
        h3=self.lin2(h3)
        return h3

if __name__=="__main__":
    save_dir = os.getcwd() + "/dataset"
    dataset = CGRDataset(save_dir=save_dir)
    cuda_g=dataset[0].to('cuda:0')
    model=Model_SAGE(len(dataset[0].ndata["x"][0]),1).double()
    print(model(dataset[0], dataset[0].ndata["x"]))