# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from itertools import chain

import networkx as nx
from typing import Any, Collection


def m_separated(graph: nx.DiGraph, x: Any, y: Any, z: Collection[Any]) -> bool:
    # Uses the Bayes-Ball algorithm as described in "Separators and Adjustment Sets in Causal Graphs:
    # Complete Criteria and an Algorithmic Framework" - van der Zander et al.
    # This implementation does not support edges without arrowhead, i.e. selection bias
    queue = [(x, neighbour) for neighbour in chain(graph.successors(x), graph.predecessors(x))]
    visited = set([])
    while queue:
        l, m = queue.pop(0)
        for n in chain(graph.successors(m), graph.predecessors(m)):
            edge = frozenset([m, n])
            if edge not in visited and _m_connected_segment(graph, l, m, n, z):
                if n == y:
                    return False
                queue.append((m, n))
                visited = visited.union({frozenset([m, n])})
    return True


def _m_connected_segment(graph: nx.DiGraph, l: Any, m: Any, n: Any, conditioning_set: Collection[Any]) -> bool:
    if m not in conditioning_set:
        if (graph.has_edge(l, m) and graph.has_edge(m, n)   # This includes bidirected edges between l and m
                or graph.has_edge(m, l)):  # and any edge between m and n
            return True
    else:  # m in conditioning set
        if graph.has_edge(l, m) and graph.has_edge(n, m):   # this includes bidirected edges between any of the nodes
            return True
    return False
