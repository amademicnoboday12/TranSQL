import onnx
from onnx import shape_inference, numpy_helper
from parameterization import (
    MatmulParam,
    ElementwiseFunctionParam,
    ElementwiseArithmeticParam,
    ReshapeParam,
    NormalizeParam
)


def get_tensor_dims(model: onnx.ModelProto) -> dict:
    """
    Build a mapping from tensor names to a list of dimension names.
    Uses graph inputs, outputs, and value_info.
    """
    dims = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value.type.HasField('tensor_type'):
            tensor_shape = value.type.tensor_type.shape
            dim_names = []
            for idx, dim in enumerate(tensor_shape.dim):
                name = f"{value.name}_dim{idx}"
                dim_names.append(name)
            if dim_names:
                dims[value.name] = dim_names
    return dims

def extract_constant_shape(model: onnx.ModelProto, name: str):
    """
    Extract constant shape from initializers (for Reshape).
    """
    for init in model.graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            return list(arr)
    return None


def traverse_graph(onnx_path: str):
    """
    Load an ONNX model, infer shapes, traverse its computation graph,
    and instantiate parameterization objects for each supported operator.
    """
    model = onnx.load(onnx_path)
    dims_map = get_tensor_dims(model)
    params = []

    for node in model.graph.node:
        op = node.op_type
        output_name = node.output[0] if node.output else None
        # Matrix multiplication / Gemm
        if op in ["MatMul", "Gemm"]:
            A, B = node.input[0], node.input[1]
            row_dim_A = dims_map[A][0]
            shared_dim_A = dims_map[A][1]
            shared_dim_B = dims_map[B][0]
            col_dim_B = dims_map[B][1]
            param = MatmulParam(A, B, row_dim_A, col_dim_B, shared_dim_A, shared_dim_B,output_name=output_name)

        # Element-wise functions
        elif op in ["Sigmoid", "Relu", "Gelu", "Swish"]:
            A = node.input[0]
            free_dims = dims_map.get(A, [])
            param = ElementwiseFunctionParam(A, free_dims, func=op.lower(),output_name=output_name)

        # Element-wise arithmetic
        elif op in ["Add", "Mul", "Sub", "Div"]:
            A, B = node.input[0], node.input[1]
            dims_A = dims_map.get(A, [])
            dims_B = dims_map.get(B, [])
            shared_dims = list(zip(dims_A, dims_B))
            func_map = {"Add": "add", "Mul": "mul", "Sub": "sub", "Div": "div"}
            param = ElementwiseArithmeticParam(A, B, shared_dims, func_map[op],output_name=output_name)

        # Reshape
        elif op == "Reshape":
            A, shape_name = node.input[0], node.input[1]
            const_shape = extract_constant_shape(model, shape_name)
            if const_shape:
                new_dims = [f"{node.name}_dim{i}" for i in range(len(const_shape))]
            else:
                new_dims = []
            param = ReshapeParam(A, new_dims, func=f"reshape({A}, {const_shape})",output_name=output_name)

        # Softmax
        elif op == "Softmax":
            A = node.input[0]
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
            tensor_dims = dims_map.get(A, [])
            row_dim = tensor_dims[0]
            col_dim = tensor_dims[1] if len(tensor_dims) > 1 else None
            free_dims = [col_dim] if col_dim else []
            shared_dims = [(row_dim, row_dim)]
            group_dims = [row_dim]
            param = NormalizeParam(A, f="exp", agg="SUM", g="div",
                                   free_dims=free_dims,
                                   shared_dims=shared_dims,
                                   group_dims=group_dims,output_name=output_name)

        # LayerNormalization / RMSNormalization
        elif op in ["LayerNormalization", "RMSNormalization"]:
            A = node.input[0]
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
            tensor_dims = dims_map.get(A, [])
            if axis < 0:
                axis = len(tensor_dims) + axis
            axis_dim = tensor_dims[axis]
            other_dims = [d for d in tensor_dims if d != axis_dim]
            shared_dims = [(axis_dim, axis_dim)]
            group_dims = other_dims
            free_dims = [axis_dim]
            param = NormalizeParam(A, f="id", agg="MEAN", g="subdiv",
                                   free_dims=free_dims,
                                   shared_dims=shared_dims,
                                   group_dims=group_dims,output_name=output_name)

        else:
            continue
        if(output_name.startswith("key_state") or output_name.startswith("value_state")):
            param.is_key_or_value = True
        if(output_name.endswith('layernorm/Mul_1_output_0')):
            param.is_skip_connection = True
        params.append(param)

    return params

# if __name__ == "__main__":
#     import sys
#     onnx_path = sys.argv[1]
#     params = traverse_graph(onnx_path)
#     for p in params:
#         print(p)