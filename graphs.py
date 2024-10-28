import pandas as pd
import numpy as np
import dgl
import torch
import networkx as nx
import os

def read_edge_in_one_graph(file):
    df = pd.read_excel(file, header=None)
    list = df.values.tolist()
    ar = []
    for i in range(len(list)):
        tuple_edge = (list[i][0], list[i][1])
        ar.append(tuple_edge)
    for i in range(len(list)):
        tuple_edge = (list[i][1], list[i][0])
        ar.append(tuple_edge)
    return ar

def read_components_in_one_graph(file):
    df=pd.read_excel(file,header=None).to_numpy()
    return df

def read_node_features_in_one_graph(file):
    df=pd.read_excel(file,header=None)
    list = df.values.tolist()
    for i in range(len(list)):
        list[i][0] = list[i][0] / 0.05
        list[i][1] = list[i][1] / 0.05
        list[i][2] = list[i][2] / 0.05
        list[i][3] = list[i][3] / 1
        list[i][4] = list[i][4] / 1
        list[i][5] = list[i][5] / 12
        list[i][6] = list[i][6] / 1
        list[i][7] = list[i][7] / 500
        list[i][8] = list[i][8] / 50
        list[i][9] = list[i][9] / 1
        list[i][10] = list[i][10] / 50000
        list[i][11] = list[i][11] / 50000
    ar = np.zeros(shape=(len(list), len(list[0]))).astype(float)
    for i in range(len(list)):
        for j in range(len(list[0])):
            ar[i, j] = list[i][j]
    return ar

def read_node_coordinate(file,comps):
    df = pd.read_excel(file, header=None).to_numpy()
    ar=np.zeros(shape=(len(comps),3))
    for i in range(len(df)):
        df[i,0]=df[i,0]/120
        df[i,1]=df[i,1]/50
        df[i,2]=df[i,2]/70
    for i in range(len(comps)):
        index1=comps[i][0]
        index2=comps[i][1]
        ar[i]=(df[index2]+df[index1])/2
    return ar


def read_node_result_in_one_graph(file):
    df=pd.read_excel(file,header=None)
    list = df.values.tolist()
    ar = np.zeros(shape=(len(list), len(list[0]))).astype(int)
    for i in range(len(list)):
        for j in range(len(list[0])):
            ar[i, j] = int(list[i][j])
    return ar

def create_one_graph(file_e,file_n_f,file_n_r,file_n_pos,file_components):
    edge = read_edge_in_one_graph(file_e)
    components=read_components_in_one_graph(file_components)
    nxg = nx.DiGraph(edge)
    g = dgl.from_networkx(nxg)
    node_f=read_node_features_in_one_graph(file_n_f)
    node_pos = read_node_coordinate(file_n_pos, components)
    node_pos_f=np.concatenate((node_pos, node_f), axis=1)
    node_result=read_node_result_in_one_graph(file_n_r)
    g.ndata['x']=torch.tensor(node_pos_f)
    g.ndata['y'] = torch.tensor(node_result)

    '''u,s,v=torch.pca_lowrank(g.ndata['x'])
    g.ndata['x']=torch.mm(g.ndata['x'],v[:,:6])'''
    return g

def origin_graph_data(root_file):
    data_files_all = os.listdir(root_file)
    origin_graphs=[]
    for i in range(len(data_files_all)):
        g=create_one_graph(root_file + "data"+str(i) + "/connect.xlsx",
                           root_file + "data"+str(i) + "/x.xlsx",
                           root_file + "data"+str(i) + "/y.xlsx",
                           root_file + "data"+str(i) + "/joint_coordinate.xlsx",
                           root_file + "data"+str(i) + "/connect_comp.xlsx")
        #print(g.ndata['x'][0])
        origin_graphs.append(g)
    return origin_graphs