from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class OperatorParam:
    """
    Base class for neural operator parameterization.
    operands: list of relation/table names for inputs.
    free_dims: dimensions retained in output (F).
    shared_dims: list of tuples mapping shared dimensions between operands (S).
    group_dims: dimensions to group by in SQL (G).
    attributes: additional operator-specific attributes.
    """
    operands: List[str]
    free_dims: List[str]
    shared_dims: List[Tuple[str, str]]
    group_dims: List[str]
    attributes: Dict[str, Any]
    is_skip_connection: bool = False
    is_key_or_value: bool = False
    output_name: str = ""

@dataclass
class MatmulParam(OperatorParam):
    """
    Parameterization for matrix multiplication:
    free_dims = []
    shared_dims = [(r_A, r_B)]
    group_dims = [m_A, n_B]
    """
    def __init__(
        self,
        A: str,
        B: str,
        row_dim_A: str,
        col_dim_B: str,
        shared_dim_A: str,
        shared_dim_B: str,
        output_name: str = ""
    ):
        operands = [A, B]
        free_dims: List[str] = []
        shared_dims: List[Tuple[str, str]] = [(shared_dim_A, shared_dim_B)]
        group_dims: List[str] = [row_dim_A, col_dim_B]
        attributes: Dict[str, Any] = {}
        super().__init__(operands, free_dims, shared_dims, group_dims, attributes,output_name=output_name)

@dataclass
class ElementwiseFunctionParam(OperatorParam):
    """
    Parameterization for element-wise functions (e.g., Sigmoid, SiLU):
    free_dims = [m_A, n_A]
    shared_dims = []
    group_dims = []
    attributes = {'func': name_of_function}
    """
    def __init__(self, A: str, free_dims: List[str], func: str,output_name: str = ""):
        operands = [A]
        shared_dims: List[Tuple[str, str]] = []
        group_dims: List[str] = []
        attributes: Dict[str, Any] = {'func': func}
        super().__init__(operands, free_dims, shared_dims, group_dims, attributes,output_name=output_name)

@dataclass
class ElementwiseArithmeticParam(OperatorParam):
    """
    Parameterization for element-wise arithmetic (e.g., A + B):
    free_dims = []
    shared_dims = [(dim_A, dim_B), ...]
    group_dims = []
    attributes = {'func': name_of_operation}
    """
    def __init__(self, A: str, B: str, shared_dims: List[Tuple[str, str]], func: str,output_name: str = ""):
        operands = [A, B]
        free_dims: List[str] = []
        group_dims: List[str] = []
        attributes: Dict[str, Any] = {'func': func}
        super().__init__(operands, free_dims, shared_dims, group_dims, attributes,output_name=output_name)

@dataclass
class ReshapeParam(OperatorParam):
    """
    Parameterization for reshape/flatten/view:
    free_dims = new_dims
    shared_dims = []
    group_dims = new_dims
    attributes = {'func': reshape_expression}
    """
    def __init__(self, A: str, new_dims: List[str], func: str,output_name: str = ""):
        operands = [A]
        shared_dims: List[Tuple[str, str]] = []
        group_dims: List[str] = new_dims
        attributes: Dict[str, Any] = {'func': func}
        super().__init__(operands, new_dims, shared_dims, group_dims, attributes,output_name=output_name)

@dataclass
class NormalizeParam(OperatorParam):
    """
    Parameterization for normalization-like operations (e.g., Softmax, LayerNorm):
    attributes = {'f': pre_fn, 'agg': aggregation_fn, 'g': post_fn}
    """
    def __init__(
        self,
        A: str,
        f: str,
        agg: str,
        g: str,
        free_dims: List[str],
        shared_dims: List[Tuple[str, str]],
        group_dims: List[str],
        output_name: str = ""
    ):
        operands = [A]
        attributes: Dict[str, Any] = {'f': f, 'agg': agg, 'g': g}
        super().__init__(operands, free_dims, shared_dims, group_dims, attributes,output_name=output_name)