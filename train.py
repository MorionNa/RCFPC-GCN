# coding:utf-8
from dataset import CGRDataset
from model import Model_SAGE

from loss import FocalLossV1
import datetime
import numpy as np
from dgl.dataloading import GraphDataLoader
import torch
import os
import pandas as pd

str_loss=""
epoch_loss=[]
time=[]
def train(model,optimizer,scheduler,device,dataset):
    #global str_loss
    global epoch_loss
    '''ratio=torch.tensor([0.7]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=ratio)
    0.8,4'''
    criterion=FocalLossV1(alpha=0.25,gamma=2)
    #criterion = focal_loss.F1Loss()
    #print(scheduler.get_last_lr())
    model.train()
    batch_loss=[]
    for data in dataset:
        cuda_g=data.to(device)
        out = model(cuda_g,cuda_g.ndata['x'])
        loss = criterion(out, cuda_g.ndata['y'].double())
        batch_loss.append(loss.item())
        #str_loss=str_loss+str(loss.item())+"\n"
        optimizer.zero_grad()
        #loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
    scheduler.step()
    epoch_loss.append(np.sum(batch_loss)/len(dataset))
    time.append(datetime.datetime.today())

def test(model,dataset,device):
    model.eval()
    with torch.no_grad():
        TP=torch.tensor([0]).to(device)
        FP=torch.tensor([0]).to(device)
        FN=torch.tensor([0]).to(device)
        total=torch.tensor([0]).to(device)
        count=0
        for data in dataset:
            cuda_g=data.to(device)
            pred=model(cuda_g,cuda_g.ndata['x'])
            sample_pred=torch.ge(pred,0.1).type(torch.int32)
            '''print(torch.sum(sample_pred))
            print(torch.sum(cuda_g.ndata['y']))'''
            #print(sample_pred)
            TP=TP+torch.sum(sample_pred*cuda_g.ndata['y'])
            FP=FP+torch.sum(sample_pred)-torch.sum(sample_pred*cuda_g.ndata['y'])
            FN=FN+torch.sum(cuda_g.ndata['y'])-torch.sum(sample_pred*cuda_g.ndata['y'])
            total=total+len(cuda_g.ndata['y'])
            '''pred_file=open('pred'+str(count)+".txt",'w')'''
            count=count+1
            '''pred_file.write(str(sample_pred.tolist()))'''
        #print(TP,FP,FN, total)
        return {'accuracy':(total-FP-FN)/total,'precision':TP/(TP+FP),'recall':TP/(TP+FN),'F1-score':2*TP/(TP+FP)*TP/(TP+FN)/(TP/(TP+FP)+TP/(TP+FN))}

def data_augment_by_rotation(dataset):
    #print(len(dataset))
    for i in range(len(dataset)):
        dataset.append(dataset[i])
        dataset[len(dataset)-1].ndata['x'][:,:3]=dataset[len(dataset)-1].ndata['x'][:,[1,0,2]]
    return dataset

def data_augment_by_transform(dataset):
    for i in range(len(dataset)):
        dataset.append(dataset[i])
        max_pos_x=torch.max(dataset[len(dataset)-1].ndata['x'][:,0])
        dataset[len(dataset)-1].ndata['x'][:,0]=-dataset[len(dataset)-1].ndata['x'][:,0]
        dataset[len(dataset) - 1].ndata['x'][:, 0]=dataset[len(dataset) - 1].ndata['x'][:, 0].add(max_pos_x)
    return dataset

def array2excel(ar_list,name_list,path):
    for i in range(len(ar_list)):
        data = pd.DataFrame(ar_list[i])
        writer = pd.ExcelWriter(path+name_list[i]+".xlsx")
        data.to_excel(writer, header=False, index=False)
        writer._save()

if __name__=="__main__":
    # 参数
    lr = 0.01
    weight_decay = 1e-6
    Max_step = 20000
    save_step = 10000
    print_step=100
    num_trainset = 0.7
    device = torch.device('cuda:0')

    # 数据集
    save_dir = os.getcwd() + "/dataset"
    dataset = CGRDataset(save_dir=save_dir)
    seed = 1234
    torch.manual_seed(seed)
    shuffle_idx=torch.randperm(len(dataset))
    dataset=[dataset[i] for i in shuffle_idx]
    train_dataset = dataset[:int(num_trainset * len(dataset))]
    data_augment_by_rotation(train_dataset)
    data_augment_by_transform(train_dataset)

    test_dataset = dataset[int(num_trainset * len(dataset)):]
    train_loader = GraphDataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = GraphDataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    # 模型
    model = Model_SAGE(len(dataset[0].ndata["x"][0]), len(dataset[0].ndata['y'][0])).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    #print(model.state_dict())
    print(datetime.datetime.today())
    print('-------------start training-------------')
    for epoch in range(Max_step):
        train(model,optimizer,scheduler,device,train_loader)
        if (epoch+1)%print_step==0:
            print(test(model,train_loader,device))
            print(test(model,test_loader,device))
            #print(str_loss)
            '''file=open('loss_count.txt','w')
            file.write(str_loss)'''
            #print(model.sage.conv1.fc_neigh.weight.grad)
        if (epoch+1)%save_step==0:
            torch.save(model, "CGR__SAGE_model_" + str(epoch + 1) + ".pth")
        print(str(epoch+1) + "/" + str(Max_step))
        if epoch == Max_step-1:
            array2excel([epoch_loss,time],['epoch_loss','time'],os.getcwd()+'/')
            print('-------------finish-------------')

