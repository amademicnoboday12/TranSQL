from typing import List, Dict, Any, Set
from parameterization import (
    OperatorParam,
    MatmulParam,
    ElementwiseFunctionParam,
    ElementwiseArithmeticParam,
    ReshapeParam,
    NormalizeParam
)

class SQLGenerator:
    """
    Generates SQL queries from parameterized neural operators.
    """
    def __init__(self):
        pass

    def generate_sql(self, param: OperatorParam) -> str:
        if isinstance(param, MatmulParam):
            return self._generate_matmul(param)
        elif isinstance(param, ElementwiseFunctionParam):
            return self._generate_elementwise_func(param)
        elif isinstance(param, ElementwiseArithmeticParam):
            return self._generate_elementwise_arith(param)
        elif isinstance(param, ReshapeParam):
            return self._generate_reshape(param)
        elif isinstance(param, NormalizeParam):
            return self._generate_normalize(param)
        elif isinstance(param, OperatorParam):
            print(f"special cases: {param}")
        else:
            raise ValueError(f"Unsupported operator parameter: {type(param)}")

    def _generate_matmul(self, p: MatmulParam) -> str:
        A, B = p.operands
        (rA, rB) = p.shared_dims[0]
        mA, nB = p.group_dims
        return f"""
        SELECT
          A.{mA} AS {mA},
          B.{nB} AS {nB},
          SUM(DOT(A.chunk, B.chunk)) AS value_{mA}_{nB}
        FROM {A} AS A
        JOIN {B} AS B
          ON A.{rA} = B.{rB}
        GROUP BY A.{mA}, B.{nB}
        """

    def _generate_elementwise_func(self, p: ElementwiseFunctionParam) -> str:
        A = p.operands[0]
        func = p.attributes['func']
        dims = p.free_dims
        dims_select = ", ".join(f"A.{d}" for d in dims)
        return f"""
        SELECT
          {dims_select},
          {func}(A.chunk) AS chunk
        FROM {A} AS A
        """

    def _generate_elementwise_arith(self, p: ElementwiseArithmeticParam) -> str:
        A, B = p.operands
        func = p.attributes['func']
        join_conditions = " AND ".join(f"A.{dA} = B.{dB}" for dA, dB in p.shared_dims)
        dims_select = ", ".join(f"A.{dA}" for dA, _ in p.shared_dims)
        return f"""
        SELECT
          {dims_select},
          A.chunk {func} B.chunk AS chunk
        FROM {A} AS A
        JOIN {B} AS B
          ON {join_conditions}
        """

    def _generate_reshape(self, p: ReshapeParam) -> str:
        A = p.operands[0]
        func = p.attributes['func']
        dims = p.group_dims
        dims_select = ", ".join(f"A.{d}" for d in dims)
        return f"""
        SELECT
          {dims_select},
          A.chunk AS chunk
        FROM {A} AS A
        -- reshape via: {func}
        """

    def _generate_normalize(self, p: NormalizeParam) -> str:
        A = p.operands[0]
        f = p.attributes['f']
        agg = p.attributes['agg']
        g = p.attributes['g']
        free = p.free_dims
        group = p.group_dims
        shared = p.shared_dims[0]
        sel1 = ", ".join(f"A.{d}" for d in free)
        grp  = ", ".join(group)
        return f"""
        WITH exp_vals AS (
          SELECT
            {sel1},
            {f}(A.chunk) AS exp_chunk
          FROM {A} AS A
        ),
        agg_vals AS (
          SELECT
            {sel1},
            {agg}(exp_chunk) AS sum_exp
          FROM exp_vals
          GROUP BY {grp}
        )
        SELECT
          {sel1}, {grp},
          {g}(exp_vals.exp_chunk, agg_vals.sum_exp) AS normalized_chunk
        FROM exp_vals
        JOIN agg_vals
          ON exp_vals.{shared[0]} = agg_vals.{shared[1]}
        """


def generate_sql_scripts(params: List[OperatorParam],
                         critical_names: Set[str]) -> List[str]:
    """
    Generate optimized SQL scripts by:
      1) emitting one SQL per param
      2) marking some as critical (they get materialized)
      3) merging the rest via CTEs
    """
    sql_text=[]
    gen = SQLGenerator()
    # build node list
    for p in params:
        sql_text.append(gen.generate_sql(p))
    
        # print(p.operands,view_name)
    # run the CTE-merging optimizer
    return sql_text
# Example usage:

