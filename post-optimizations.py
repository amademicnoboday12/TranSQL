from typing import List, Dict, Any, Set
from config import *
from parameterization import (
    MatmulParam,
    OperatorParam
)
from code_generator import generate_sql_scripts

def topo_sort(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Kahn’s algorithm over our node dicts.
    Only treats deps that match another node’s name.
    Returns a list of node dicts in topological order.
    """
    from collections import defaultdict, deque

    # Map view name → node dict
    name2node = {n['name']: n for n in nodes}

    # Initialize indegree and children for each node
    indegree = {name: 0 for name in name2node}
    children = {name: [] for name in name2node}

    # Build graph: only consider deps that are other view names
    for n in nodes:
        for dep in n['deps']:
            if dep in name2node:
                indegree[n['name']] += 1
                children[dep].append(n['name'])

    # Start with all zero-indegree nodes
    q = deque(name for name, deg in indegree.items() if deg == 0)
    ordered: List[Dict[str, Any]] = []

    while q:
        name = q.popleft()
        node = name2node[name]
        ordered.append(node)
        for child_name in children[name]:
            indegree[child_name] -= 1
            if indegree[child_name] == 0:
                q.append(child_name)

    return ordered


def merge_cte_chain(chain: List[Dict[str, Any]]) -> str:
    cte_defs = []
    for n in chain:
        # strip trailing semicolons
        body = n['sql'].strip().rstrip(';')
        cte_defs.append(f"{n['name']} AS (\n{body}\n)")
    with_clause = "WITH\n  " + ",\n  ".join(cte_defs)
    # finally select from the last view
    return f"""{with_clause}
SELECT * FROM {chain[-1]['name']};"""

def optimize_cte_queries(nodes: List[Dict[str, Any]]) -> List[str]:
    sorted_nodes = topo_sort(nodes)
    final_sqls: List[str] = []
    cte_chain: List[Dict[str, Any]] = []
    for node in sorted_nodes:
        if node['is_critical']:
            if cte_chain:
                final_sqls.append(merge_cte_chain(cte_chain))
                cte_chain = []
            # materialize critical node as‐is
            final_sqls.append(node['sql'].strip())
        else:
            cte_chain.append(node)

    if cte_chain:
        final_sqls.append(merge_cte_chain(cte_chain))

    return final_sqls


def optimize_matmul_r2c(p: MatmulParam, num_chunk) -> str:
    A, B = p.operands
    (rA, rB) = p.shared_dims[0]
    mA, nB = p.group_dims
    N = num_chunk

    # build pivot expressions for A and B
    pivot_A = ",\n    ".join(
        f"MAX(CASE WHEN A.{rA} = {i} THEN A.chunk END) AS chunk_{i}"
        for i in range(N)
    )
    pivot_B = ",\n    ".join(
        f"MAX(CASE WHEN B.{rB} = {i} THEN B.chunk END) AS chunk_{i}"
        for i in range(N)
    )

    # sum of DOTs across each chunk column
    dot_sum = " +\n    ".join(
        f"DOT(Ap.chunk_{i}, Bp.chunk_{i})"
        for i in range(N)
    )

    return f"""
    WITH
    A_pivot AS (
        SELECT
        A.{mA},
        {pivot_A}
        FROM {A} AS A
        GROUP BY A.{mA}
    ),
    B_pivot AS (
        SELECT
        B.{nB},
        {pivot_B}
        FROM {B} AS B
        GROUP BY B.{nB}
    )
    SELECT
        Ap.{mA}   AS {mA},
        Bp.{nB}   AS {nB},
        {dot_sum} AS value_{mA}_{nB}
    FROM A_pivot AS Ap
    CROSS JOIN B_pivot AS Bp;
    """

def optimizations(params: List[OperatorParam], sql_text: List[str],  
                          critical_names: Set[str]) -> List[str]:

    for i,p in enumerate(params):
        if isinstance(p, MatmulParam):
            # optimize MatMul to R2C
            sql_text[i] = optimize_matmul_r2c(p,NUM_CHUNK)

    nodes: List[Dict[str, Any]] = []
    for p,sql in zip(params,sql_text):
        view_name = p.output_name                # assumes each param has a unique .name
        nodes.append({
            'name':      view_name,
            'sql':       sql,
            'is_critical': view_name in critical_names,
            'deps':      list(p.operands),  # operand names are the dependencies
        })
    optimized_sqls= optimize_cte_queries(nodes)
    return optimized_sqls


from onnx_graph_traversal import traverse_graph

params = traverse_graph('example_model.onnx')
# e.g. you decide that all key/value and skip‐conn params get materialized:
critical = { p.output_name for p in params if p.is_key_or_value or p.is_skip_connection }
# print(f"Critical params: {critical}")
sql_text = generate_sql_scripts(params, critical)
with open('sql_scripts_concept.sql', 'w') as f:
    for sql in sql_text:
        f.write(sql + "\n\n")
scripts=optimizations(params, sql_text, critical)
with open('optimized_sql_scripts_concept.sql', 'w') as f:
    for script in scripts:
        f.write(script + "\n\n")
print(len(scripts), "SQL scripts generated.")