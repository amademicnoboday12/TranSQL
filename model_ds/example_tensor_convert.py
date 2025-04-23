model_name="deepseek-ai/deepseek-moe-16b-base"
import torch
from transformers import AutoModel
import pandas as pd
import numpy as np
import math

model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
state_dict = model.state_dict()



#attention
data=[]
for i in range(28):
    tensor_q=state_dict[f'layers.{i}.self_attn.q_proj.weight'].cpu().numpy().reshape((16,128,32,64))
    for h in range(16):
        for r in range(128):
            for c in range(32):
                data.append([0,i,h,r,c,tensor_q[r,c,:]])
    tensor_k=state_dict[f'layers.{i}.self_attn.k_proj.weight'].cpu().numpy().reshape((16,128,32,64))
    for h in range(16):
        for r in range(128):
            for c in range(32):
                data.append([1,i,h,r,c,tensor_k[r,c,:]])
    tensor_v=state_dict[f'layers.{i}.self_attn.v_proj.weight'].cpu().numpy().reshape((16,128,32,64))
    for h in range(16):
        for r in range(2048):
            for c in range(32):
                data.append([2,i,h,r,c,tensor_v[r,c,:]])
df=pd.DataFrame(data,columns=['type','layer','head_id','row_tile','col_tile','chunk'])
df.to_parquet('deepseek/attentions.parquet')


#Wo
data=[]
for i in range(28):
    tensor_o=state_dict[f'layers.{i}.self_attn.out_proj.weight'].cpu().numpy().T.reshape((2048,16,128))
    for r in range(2048):
        for h in range(16):
                data.append([i,r,h,tensor_o[r,h,:]])
df=pd.DataFrame(data,columns=['layer','row_tile','head_id','chunk'])
df.to_parquet('deepseek/attentions_o.parquet')

#mlp_gate_proj, up_proj
data=[]
tensor=state_dict[f'layers.0.mlp.gate_proj.weight'].cpu().numpy().reshape((10944,32,64))
for r in range(10944):
    for c in range(32):
            data.append([0,r,c,tensor[r,c,:]])
tensor=state_dict[f'layers.0.mlp.up_proj.weight'].cpu().numpy().reshape((10944,32,64))
for r in range(10944):
    for c in range(32):
            data.append([1,r,c,tensor[r,c,:]])
df=pd.DataFrame(data,columns=['type','row_tile','col_tile','chunk'])
df.to_parquet('deepseek/mlp_gate_up_proj.parquet')

#mlp down_proj
data=[]
tensor=state_dict[f'layers.0.mlp.down_proj.weight'].cpu().numpy().reshape((2048,171,64))
for r in range(2048):
    for c in range(171):
            data.append([r,c,tensor[r,c,:]])
df=pd.DataFrame(data,columns=['row_tile','col_tile','chunk'])
df.to_parquet('deepseek/mlp_down_proj.parquet')

#input_attention_norm, post_attention_norm, final_norm
data=[]
for i in range(28):
    tensor1=state_dict[f'layers.{i}.input_layernorm.weight'].cpu().numpy().reshape((32,64))
    for r in range(32):
            data.append([0,i,r,tensor1[r]])
    tensor2=state_dict[f'layers.{i}.post_attention_layernorm.weight'].cpu().numpy().reshape((32,64))
    for r in range(32):
            data.append([1,i,r,tensor2[r]])
tensor3=state_dict[f'norm.weight'].cpu().numpy().reshape((32,64))
for r in range(32):
    data.append([2,0,r,tensor3[r]])
df=pd.DataFrame(data,columns=['type','layer','row_tile','chunk'])
df.to_parquet('deepseek/attention_norm.parquet')

#expert gate_proj, up_proj, shared_expert gate_proj, up_proj
data=[]
for i in range(1,28):
    for j in range(64):
        tensor1=state_dict[f'layers.{i}.mlp,experts.{j},gate_proj.weight'].cpu().numpy().reshape((1408,32,64))
        for r in range(1408):
            for c in range(32):
                data.append([0,i,j,r,c,tensor1[r,c,:]])
        tensor2=state_dict[f'layers.{i}.mlp.experts.{j}.up_proj.weight'].cpu().numpy().reshape((1408,32,64))
        for r in range(1408):
            for c in range(64):
                data.append([1,i,j,r,c,tensor2[r,c,:]])
    tensor3=state_dict[f'layers.{i}.mlp.shared_experts.gate_proj.weight'].cpu().numpy().reshape((2816,32,64))
    for r in range(2816):
        for c in range(32):
            data.append([0,i,64,r,c,tensor3[r,c,:]])
    tensor4=state_dict[f'layers.{i}.mlp.shared_experts.up_proj.weight'].cpu().numpy().reshape((2816,32,64))
    for r in range(2816):
        for c in range(32):
            data.append([1,i,64,r,c,tensor4[r,c,:]])
df=pd.DataFrame(data,columns=['type','layer','expert_id','row_tile','col_tile','chunk'])
df.to_parquet('deepseek/experts_gate_up_proj.parquet')

#expert down_proj
data=[]
for i in range(1,28):
    for j in range(64):
        tensor=state_dict[f'layers.{i}.mlp.experts.{j}.down_proj.weight'].cpu().numpy().reshape((2048,22,64))
        for r in range(2048):
            for c in range(22):
                data.append([i,j,r,c,tensor[r,c,:]])
    tensor1=state_dict[f'layers.{i}.mlp.shared_experts.down_proj.weight'].cpu().numpy().reshape((2048,44,64))
    for r in range(2048):
        for c in range(44):
            data.append([i,64,r,c,tensor1[r,c,:]])
df=pd.DataFrame(data,columns=['layer','expert_id','row_tile','col_tile','chunk'])
df.to_parquet('deepseek/expert_down_proj.parquet')

#gate_weight
data=[]
for i in range(1,28):
    tensor=state_dict[f'layers.{i}.mlp.gate.weight'].cpu().numpy().reshape((64,32,64))
    for r in range(64):
        for c in range(32):
            data.append([i,r,c,tensor[r,c,:]])
df=pd.DataFrame(data,columns=['layer','row_tile','col_tile','chunk'])
df.to_parquet('deepseek/gate_weight.parquet')

