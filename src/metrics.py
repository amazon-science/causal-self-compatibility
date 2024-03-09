# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, List

import networkx as nx
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def edge_precision(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = edge_lists(gt, g_hat)
    return precision_score(true, est)


def edge_recall(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = edge_lists(gt, g_hat)
    return recall_score(true, est)


def edge_f1(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = edge_lists(gt, g_hat)
    return f1_score(true, est)


def edge_fpr(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    fp = 0
    n = 0
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        if not gt.has_edge(x, y):
            n += 1
            if g_hat.has_edge(x, y):
                fp += 1
    return float(fp) / n if n != 0 else 1.


def edge_lists(gt: nx.DiGraph, g_hat: nx.DiGraph) -> Tuple[List[bool], List[bool]]:
    true_edges = []
    est_edges = []
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        true_edges.append(gt.has_edge(x, y))
        est_edges.append(g_hat.has_edge(x, y))
    return true_edges, est_edges


def skeleton_tpr(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    tp = 0
    p = 0
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        if gt.has_edge(x, y) or gt.has_edge(y, x):
            p += 1
            if g_hat.has_edge(x, y) or g_hat.has_edge(y, x):
                tp += 1
    return float(tp) / p if p != 0 else 1.


def skeleton_fpr(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    fp = 0
    n = 0
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        if not gt.has_edge(x, y) and not gt.has_edge(y, x):
            n += 1
            if g_hat.has_edge(x, y) or g_hat.has_edge(y, x):
                fp += 1
    return float(fp) / n if n != 0 else 1.


def skeleton_precision(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    tp = 0
    fp = 0
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        if g_hat.has_edge(x, y) or g_hat.has_edge(y, x):
            if gt.has_edge(x, y) or gt.has_edge(y, x):
                tp += 1
            else:
                fp += 1
    return float(tp) / (fp + tp) if fp + tp != 0 else 1.


def skeleton_f1(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    ppv = skeleton_precision(gt, g_hat)
    tpr = skeleton_tpr(gt, g_hat)
    return 2 * ppv * tpr / (ppv + tpr)


def avg_degree(_: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    return float(np.mean([g_hat.degree[node] for node in g_hat.nodes]))


def avg_gt_degree(gt: nx.DiGraph, _: nx.DiGraph) -> float:
    return float(np.mean([gt.degree[node] for node in gt.nodes]))
