import datetime
import torch
import os
import pandas as pd
from model import Model_SAGE
from graphs import create_one_graph

def array2excel(ar_list,name_list,path):
    for i in range(len(ar_list)):
        data = pd.DataFrame(ar_list[i])
        writer = pd.ExcelWriter(path+name_list[i]+".xlsx")
        data.to_excel(writer, header=False, index=False)
        writer._save()

def test(model,dataset,device,files):
    model.eval()
    with torch.no_grad():
        count=0
        for i in range(len(dataset)):
            cuda_g=dataset[i].to(device)
            pred=model(cuda_g,cuda_g.ndata['x'])
            sample_pred=torch.ge(pred,0.1).type(torch.int32)
            #print(sample_pred)
            pred_data = pd.DataFrame(sample_pred)
            writer = pd.ExcelWriter("./output/pred_"+files[i]+'.xlsx')
            pred_data.to_excel(writer, header=False, index=False)
            writer._save()
            count=count+1

if __name__=="__main__":
    path=os.getcwd()
    device = torch.device('cpu')
    dataset=[]
    list_input=os.listdir(path+'/input')
    files=[]
    for file in list_input:
        data = create_one_graph(path+'/input/' +file + "/connect.xlsx",
                                path+'/input/' +file + "/x.xlsx",
                                path+'/input/' +file + "/y.xlsx",
                                path+'/input/' +file + "/joint_coordinate.xlsx",
                                path+'/input/' +file + "/connect_comp.xlsx")
        dataset.append(data)
        files.append(file)
    model=Model_SAGE(len(dataset[0].ndata["x"][0]), len(dataset[0].ndata['y'][0])).double().to(device)
    model_state_dict=torch.load(path+'/model/CGR_SAGE_20000.pth')
    model.load_state_dict(model_state_dict)

    real_y = dataset[0].ndata['y']
    print(datetime.datetime.today())
    test(model, dataset, device,files)
    print(datetime.datetime.today())
