import onnx
from typing import List, Dict
from enum import Enum
import onnx.numpy_helper as numpy_helper
import numpy as np

onnx_file = 'your_model_with_injected_shapes1.onnx'
model = onnx.load(onnx_file)
node_map={}
shape_info={}
for init in model.graph.initializer:
    node_map[init.name]=init
for inp in model.graph.input:
    node_map[inp.name]=inp
for outp in model.graph.output:
    node_map[outp.name]=outp
for node in model.graph.node:
    node_map[node.name]=node
for v in model.graph.value_info:
    shape_info[v.name]=[x.dim_value for x in v.type.tensor_type.shape.dim]
print(shape_info)

# for vi in model.graph.value_info:
#     if vi.name == '/model/Expand_output_0':
#         # Update existing
#         vi.type.tensor_type.elem_type = 1
#         for i in range(4):
#             vi.type.tensor_type.shape.dim.pop()
#         shape=[1,1,18,19]
#         for d in shape:
#             vi.type.tensor_type.shape.dim.add().dim_value = d
# model_with_known_shapes = onnx.shape_inference.infer_shapes(model)
# onnx.save(model_with_known_shapes, "your_model_with_injected_shapes1.onnx")
