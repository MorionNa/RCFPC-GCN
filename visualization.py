import math
import os

import bpy
import numpy as np
import pandas as pd

def read_file(path):
    file=pd.read_excel(path,header=None)
    return file.values

def component_size(comp,n_coo,feats):
    size=np.zeros(shape=(len(comp),3))
    for i in range(len(comp)):
        size[i,0]=feats[i,3]
        size[i,1]=feats[i,4]
        size[i,2]=np.linalg.norm(n_coo[comp[i,1]]-n_coo[comp[i,0]])
    return size

def component_rotation(comp,n_coo):
    angle = np.zeros(shape=(len(comp), 3)).astype(float)
    for i in range(len(comp)):
        vec_real = (n_coo[comp[i, 1]] - n_coo[comp[i, 0]]) / np.linalg.norm(n_coo[comp[i, 1]] - n_coo[comp[i, 0]])
        vec_virtual = np.array([0.0, 0.0, 1.0])
        vec_real_proj = np.zeros(shape=(3, 3))
        vec_virtual_proj = np.zeros(shape=(3, 3))
        for j in range(len(angle[0])):
            vec_real_proj[j] = vec_real
            vec_real_proj[j, j] = 0
            vec_virtual_proj[j] = vec_virtual
            vec_virtual_proj[j, j] = 0
            if np.any(vec_virtual_proj[j]):
                vec_virtual_proj[j] = vec_virtual_proj[j] / np.linalg.norm(vec_virtual_proj[j])
            if np.any(vec_real_proj[j]):
                vec_real_proj[j] = vec_real_proj[j] / np.linalg.norm(vec_real_proj[j])
            cos_angle = np.dot(vec_real_proj[j], vec_virtual_proj[j])
            sin_angle = np.linalg.norm(np.cross(vec_real_proj[j], vec_virtual_proj[j]))
            angle[i,j] = np.arctan2(sin_angle, cos_angle)
    return angle

def component_location(comp,n_coo):
    loc=np.zeros(shape=(len(comp),3))
    for i in range(len(comp)):
        loc[i]=(n_coo[comp[i,1]]+n_coo[comp[i,0]])/2
    return loc

def component_object_create(size,loc,angle,fail,init):
    material_fail = bpy.data.materials.new('red')
    material_fail.use_nodes = True
    r_BSDF = material_fail.node_tree.nodes["Principled BSDF"]
    r_BSDF.inputs[0].default_value = (1, 0, 0, 1)
    material_init=bpy.data.materials.new('green')
    material_init.use_nodes = True
    g_BSDF = material_init.node_tree.nodes["Principled BSDF"]
    g_BSDF.inputs[0].default_value = (0, 1, 0, 1)
    for i in range(len(size)):
        #name = str("component:" + str(i))
        obj = bpy.ops.mesh.primitive_cube_add(location=loc[i], scale=size[i],rotation=angle[i])
        if fail[i]==1 and init[i]==0:
            material = bpy.data.materials['red']
            bpy.context.object.data.materials.append(material)
        if init[i]==1:
            material = bpy.data.materials['green']
            bpy.context.object.data.materials.append(material)

if __name__=="__main__":
    print(os.path.dirname(os.getcwd()))
    input_path=os.getcwd()+'/input/'
    output_path=os.getcwd()+'/output/'
    input_files = os.listdir(input_path)
    connect_comp=read_file(input_path+input_files[0]+'/connect_comp.xlsx')
    node_coordinate=read_file(input_path+input_files[0]+'/joint_coordinate.xlsx')
    comp_feats=read_file(input_path+input_files[0]+'/x.xlsx')
    comp_size=component_size(connect_comp,node_coordinate,comp_feats)
    comp_location=component_location(connect_comp,node_coordinate)
    comp_rotation=component_rotation(connect_comp,node_coordinate)
    comp_fail=read_file(output_path+"pred_"+input_files[0]+'.xlsx')
    component_object_create(comp_size,comp_location,comp_rotation,comp_fail,comp_feats[:,9])
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.delete()